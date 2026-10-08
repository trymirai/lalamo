from dataclasses import dataclass, replace
from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_or_tracer
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)


@dataclass(frozen=True)
class RowStackSpec(WeightMatrixSpec):
    parts: tuple[tuple[int, WeightMatrixSpec], ...]

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
        if not is_dummy_or_tracer(weights):
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
    parts: tuple[WeightMatrix, ...]

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
