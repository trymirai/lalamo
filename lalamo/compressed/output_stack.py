from dataclasses import dataclass, replace
from itertools import accumulate, pairwise
from typing import Self

import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
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
class OutputStackPart:
    rows: int
    spec: WeightMatrixSpec


@dataclass(frozen=True)
class OutputStackSpec(WeightMatrixSpec):
    parts: tuple[OutputStackPart, ...]

    def __post_init__(self) -> None:
        if not self.parts or any(part.rows <= 0 for part in self.parts):
            raise ValueError("Output stacks require parts with positive row counts")

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
        if sum(part.rows for part in self.parts) != weights.shape[0]:
            raise ValueError("Output-stack part rows must sum to the output dimension")
        boundaries = pairwise(accumulate((part.rows for part in self.parts), initial=0))
        parts = tuple(
            part.spec.compress(
                weights[start:end],
                sharding_config=sharding_config,
                is_sharded=is_sharded,
            )
            for (start, end), part in zip(boundaries, self.parts, strict=True)
        )
        return OutputStackMatrix(self, sharding_config, is_sharded, parts)


class OutputStackMatrix(WeightMatrix[OutputStackSpec]):
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
        assert not transposed, "Output stacks are fused projections, never tied embeddings"
        return jnp.concatenate(
            tuple(part.dot(vector, keychain=keychain, forward_pass_config=forward_pass_config) for part in self.parts)
        )
