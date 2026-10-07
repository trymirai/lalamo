from dataclasses import dataclass, replace
from typing import Self

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_array
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

from .lattice import LatticeMatrix, LatticeSpec
from .qtip_gaussian import QtipGaussianMatrix, QtipGaussianSpec


@dataclass(frozen=True)
class RowStackSpec(WeightMatrixSpec):
    parts: tuple[tuple[int, QtipGaussianSpec | LatticeSpec], ...]

    def __post_init__(self) -> None:
        assert all(isinstance(spec, QtipGaussianSpec) or spec.layout == Layout.OUTPUT_INPUT for _, spec in self.parts)

    def compress(
        self,
        weights: Float[Array, "out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "RowStackMatrix":
        if not is_dummy_array(weights):
            raise ValueError("Row stacks must be constructed from existing matrices")
        assert sum(rows for rows, _ in self.parts) == weights.shape[0]
        initializer = EmptyInitializer(weights.dtype, sharding_config)
        parts = tuple(
            spec.compress(
                initializer.zeros((rows, weights.shape[1])),
                sharding_config=sharding_config,
                is_sharded=is_sharded,
            )
            for rows, spec in self.parts
        )
        return RowStackMatrix(spec=self, sharding_config=sharding_config, is_sharded=is_sharded, parts=parts)


class RowStackMatrix(WeightMatrix[RowStackSpec]):
    parts: tuple[QtipGaussianMatrix | LatticeMatrix, ...]

    def __check_init__(self) -> None:
        assert tuple((part.shape[0], part.spec) for part in self.parts) == self.spec.parts
        assert all(part.shape[1] == self.parts[0].shape[1] and part.dtype == self.dtype for part in self.parts)

    @property
    def shape(self) -> tuple[int, int]:
        return sum(part.shape[0] for part in self.parts), self.parts[0].shape[1]

    @property
    def dtype(self) -> DTypeLike:
        return self.parts[0].dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, parts=tuple(part.astype(dtype) for part in self.parts))

    def export(self) -> ExportResults:
        exported = super().export()
        parts = tuple(part for part in self.parts if isinstance(part, QtipGaussianMatrix))
        if not parts or len(parts) != len(self.parts):
            return exported

        signs, small_q = parts[0].signs, parts[0].small_q
        if any(
            not np.array_equal(part.signs, signs) or not np.array_equal(part.small_q, small_q) for part in parts[1:]
        ):
            raise ValueError("QTIP row stack parts must share signs and small_q")

        arrays = dict(exported.arrays)
        for index in range(len(parts)):
            del arrays[f"parts.{index}.signs"], arrays[f"parts.{index}.small_q"]
        arrays["signs"], arrays["small_q"] = signs, small_q
        return ExportResults(arrays, exported.metadata)

    def load_exported(self, exported_data: ExportResults, *, prefix: ParameterPath | None = None) -> Self:
        prefix = prefix or ParameterPath()
        if self.parts and all(isinstance(part, QtipGaussianMatrix) for part in self.parts):
            arrays = dict(exported_data.arrays)
            signs_path, small_q_path = prefix / "signs", prefix / "small_q"
            if signs_path in arrays and small_q_path in arrays:
                signs, small_q = arrays.pop(signs_path), arrays.pop(small_q_path)
                for index in range(len(self.parts)):
                    part_path = prefix / "parts" / index
                    arrays[part_path / "signs"], arrays[part_path / "small_q"] = signs, small_q
                exported_data = ExportResults(arrays, exported_data.metadata)

        loaded = super().load_exported(exported_data, prefix=prefix)
        assert isinstance(loaded, RowStackMatrix)
        return loaded

    def decompress(self) -> Array:
        return jnp.concatenate(tuple(part.decompress() for part in self.parts), axis=0)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec().compress(
            self.decompress(), sharding_config=self.sharding_config, is_sharded=self.is_sharded
        )

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Array:
        assert not transposed, "Row stacks are fused projections, never tied embeddings"
        return jnp.concatenate(
            tuple(part.dot(vector, keychain=keychain, forward_pass_config=forward_pass_config) for part in self.parts)
        )
