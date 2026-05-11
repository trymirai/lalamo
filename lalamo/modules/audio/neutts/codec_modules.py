from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import prod
from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array
from lalamo.modules.common import LalamoModule

__all__ = [
    "NeuCodecFSQ",
    "NeuCodecFSQConfig",
]


@dataclass(frozen=True)
class NeuCodecFSQConfig:
    levels: tuple[int, ...]
    precision: DTypeLike

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("NeuCodec FSQ levels must be non-empty.")
        if any(level <= 1 for level in self.levels):
            raise ValueError("NeuCodec FSQ levels must all be greater than 1.")

    @property
    def codebook_dim(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.levels)

    def empty(self) -> "NeuCodecFSQ":
        levels = jnp.asarray(self.levels, dtype=jnp.int32)
        basis = jnp.cumprod(jnp.concatenate([jnp.asarray([1], dtype=jnp.int32), levels[:-1]]))
        return NeuCodecFSQ(
            config=self,
            levels=levels,
            basis=basis,
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecFSQ":
        return self.empty()


class NeuCodecFSQ(LalamoModule[NeuCodecFSQConfig]):
    levels: Int[Array, " codebook_dim"]
    basis: Int[Array, " codebook_dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def indices_to_level_indices(
        self,
        indices: Int[Array, "batch tokens"],
    ) -> Int[Array, "batch tokens codebook_dim"]:
        return (indices[..., None] // self.basis) % self.levels

    def indices_to_codes(
        self,
        indices: Int[Array, "batch tokens"],
    ) -> Float[Array, "batch tokens codebook_dim"]:
        level_indices = self.indices_to_level_indices(indices)
        half_width = self.levels // 2
        return (level_indices - half_width).astype(self.config.precision) / half_width.astype(self.config.precision)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "levels": self.levels,
            "basis": self.basis,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            levels=require_array(weights["levels"]),
            basis=require_array(weights["basis"]),
        )
