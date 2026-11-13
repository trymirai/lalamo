from dataclasses import dataclass, replace
from itertools import accumulate, pairwise
from typing import Self

import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)


def _stored_layout(spec: WeightMatrixSpec) -> Layout:
    # A hybrid stores its quantized matrix's layout.
    inner = getattr(spec, "quantization_spec", spec)
    return getattr(inner, "layout", Layout.OUTPUT_INPUT)


@dataclass(frozen=True)
class OutputStackPart:
    rows: int
    spec: WeightMatrixSpec


@dataclass(frozen=True)
class OutputStackSpec(WeightMatrixSpec):
    parts: tuple[OutputStackPart, ...]

    def __post_init__(self) -> None:
        if not self.parts or any(part.rows <= 0 for part in self.parts):
            raise ValueError("Output stacks require parts with positive row counts")
        if len({_stored_layout(part.spec) for part in self.parts}) != 1:
            raise ValueError("Output stacks require parts of one layout")

    @property
    def layout(self) -> Layout:
        # Parts stack along their first stored axis: the output rows of a projection or, when they are stored
        # input-output, the vocabulary rows of an embedding table.
        return _stored_layout(self.parts[0].spec)

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
        if not isinstance(weights, Tracer):
            raise NotImplementedError("Output stacks are loaded from saved parameters, not fitted")
        stacks_outputs = self.layout == Layout.OUTPUT_INPUT
        if sum(part.rows for part in self.parts) != weights.shape[0 if stacks_outputs else 1]:
            raise ValueError("Output-stack part rows must sum to the stacked dimension")
        boundaries = pairwise(accumulate((part.rows for part in self.parts), initial=0))
        parts = tuple(
            part.spec.compress(
                weights[start:end] if stacks_outputs else weights[:, start:end],
                sharding_config=sharding_config,
                is_sharded=is_sharded,
            )
            for (start, end), part in zip(boundaries, self.parts, strict=True)
        )
        return OutputStackMatrix(self, sharding_config, is_sharded, parts)


class OutputStackMatrix(EmbeddingMatrix[OutputStackSpec]):
    parts: tuple[WeightMatrix, ...]

    def __check_init__(self) -> None:
        assert tuple(OutputStackPart(part.shape[0], part.spec) for part in self.parts) == self.spec.parts
        assert all(
            part.shape[1] == self.parts[0].shape[1] and part.dtype == self.parts[0].dtype for part in self.parts
        )

    @property
    def shape(self) -> tuple[int, int]:
        return sum(part.shape[0] for part in self.parts), self.parts[0].shape[1]

    @property
    def dtype(self) -> DTypeLike:
        return self.parts[0].dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, parts=tuple(part.astype(dtype) for part in self.parts))

    def decompress(self) -> Array:
        axis = 0 if self.spec.layout == Layout.OUTPUT_INPUT else 1
        return jnp.concatenate(tuple(part.decompress() for part in self.parts), axis=axis)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(
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
        result = None
        boundaries = pairwise(accumulate((part.shape[0] for part in self.parts), initial=0))
        for (start, end), part in zip(boundaries, self.parts, strict=True):
            if not isinstance(part, EmbeddingMatrix):
                raise TypeError(f"Cannot look rows up in a {type(part).__name__}")
            # Every part is read at a clipped index; a row keeps the read of the last part that starts at or before it.
            rows = part.lookup_embedding(
                jnp.clip(row_index - start, 0, end - start - 1),
                keychain=keychain,
                dtype=dtype,
                forward_pass_config=forward_pass_config,
            )
            result = rows if result is None else jnp.where((row_index >= start)[..., None], rows, result)
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
        assert self.spec.layout == Layout.OUTPUT_INPUT, "A stack of table rows is looked up, not multiplied"
        return jnp.concatenate(
            tuple(part.dot(vector, keychain=keychain, forward_pass_config=forward_pass_config) for part in self.parts)
        )
