from dataclasses import dataclass, replace
from typing import Self

import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.exportable import ExportResults
from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array, supports_dummy_arrays
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    ShapeDtypeSpec,
    WeightMatrix,
    WeightMatrixSpec,
)


def stored_layout(spec: WeightMatrixSpec) -> Layout:
    # A hybrid stores its quantized matrix's layout; a stack stores its parts'.
    inner = getattr(spec, "quantization_spec", spec)
    return getattr(inner, "layout", Layout.OUTPUT_INPUT)


@dataclass(frozen=True)
class OutputStackSpec(WeightMatrixSpec):
    # Each part saves its own spec, and the stack saves the row count of each part.

    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "OutputStackMatrix":
        # Only a placeholder of the whole shape is built; load_exported splits it into the saved parts.
        if not isinstance(weights, Tracer):
            raise NotImplementedError("Output stacks are loaded from saved parameters, not fitted")
        placeholder = FullPrecisionSpec().compress(weights, sharding_config=sharding_config, is_sharded=is_sharded)
        return OutputStackMatrix(self, sharding_config, is_sharded, (placeholder,))


class OutputStackMatrix(EmbeddingMatrix[OutputStackSpec]):
    # Parts stack along their first stored axis: the output rows of a projection, or, when the parts are stored
    # input-output, the vocabulary rows of an embedding table, which is then looked up part by part.
    parts: tuple[WeightMatrix, ...]

    @property
    def layout(self) -> Layout:
        return stored_layout(self.parts[0].spec)

    def __check_init__(self) -> None:
        assert self.parts
        assert all(part.shape[1] == self.parts[0].shape[1] and part.dtype == self.dtype for part in self.parts)

    def export(self) -> ExportResults:
        arrays, metadata = super().export()
        part_rows = jnp.asarray([part.shape[0] for part in self.parts], dtype=jnp.int32)
        return ExportResults({**arrays, "part_rows": part_rows}, metadata)

    def load_exported(self, exported_data: ExportResults, *, prefix: ParameterPath | None = None) -> Self:
        if prefix is None:
            prefix = ParameterPath()
        # The placeholder is output-input: a table's vocabulary is its input axis.
        output_dim, input_dim = self.parts[0].shape
        layout = stored_layout(WeightMatrixSpec.from_json(exported_data.metadata[prefix / "parts" / 0 / "spec"]))
        sharding = self.sharding_config.resolve_sharding((None, None))
        parts = tuple(
            ShapeDtypeSpec()
            .compress(
                dummy_array(
                    (rows, input_dim) if layout == Layout.OUTPUT_INPUT else (output_dim, rows), self.dtype, sharding
                ),
                sharding_config=self.sharding_config,
                is_sharded=self.is_sharded,
            )
            .load_exported(exported_data, prefix=prefix / "parts" / index)
            for index, rows in enumerate(exported_data.arrays[prefix / "part_rows"].tolist())
        )
        return replace(self, parts=parts)

    @property
    def shape(self) -> tuple[int, int]:
        return sum(part.shape[0] for part in self.parts), self.parts[0].shape[1]

    @property
    def dtype(self) -> DTypeLike:
        return self.parts[0].dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, parts=tuple(part.astype(dtype) for part in self.parts))

    def decompress(self) -> Array:
        axis = 0 if self.layout == Layout.OUTPUT_INPUT else -1
        return jnp.concatenate(tuple(part.decompress() for part in self.parts), axis=axis)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(self.layout).compress(
            self.decompress(), sharding_config=self.sharding_config, is_sharded=self.is_sharded
        )

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        keychain: Keychain,
        dtype: DTypeLike | None = None,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> Float[Array, "*batch out_channels"]:
        row_index = jnp.asarray(row_index)
        result, start = None, 0
        for part, part_keychain in zip(self.parts, keychain.split(len(self.parts)), strict=True):
            if not isinstance(part, EmbeddingMatrix):
                raise TypeError(f"Cannot look rows up in a {type(part).__name__}.")
            rows = part.shape[0]
            # Every part is read at a clipped index, and each row keeps the read of the part that holds it.
            found = part.lookup_embedding(
                jnp.clip(row_index - start, 0, rows - 1),
                keychain=part_keychain,
                dtype=dtype,
                forward_pass_config=forward_pass_config,
            )
            inside = (row_index >= start) & (row_index < start + rows)
            result = found if result is None else jnp.where(inside[..., None], found, result)
            start += rows
        assert result is not None
        return result

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Array:
        assert not transposed, "Output stacks are fused projections, never tied embeddings"
        return jnp.concatenate(
            tuple(part.dot(vector, keychain=keychain, forward_pass_config=forward_pass_config) for part in self.parts)
        )
