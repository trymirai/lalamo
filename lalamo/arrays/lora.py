from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as jr

from lalamo.common import dummy_array, is_abstract_array

from .base import ArrayForwardPassConfig, CompressedArray

if TYPE_CHECKING:
    from jaxtyping import Array, Float, PRNGKeyArray


class LoraArray(CompressedArray):
    down: Float[Array, "... out_channels rank"]
    up: Float[Array, "... rank in_channels"]

    @property
    def is_abstract(self) -> bool:
        return is_abstract_array(self.down)

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:  # noqa: ARG002, B008
        if self.is_abstract:
            *leading_dims, out_channels, _rank = self.down.shape
            _leading_dims2, _rank2, in_channels = self.up.shape
            return dummy_array((*leading_dims, out_channels, in_channels), self.down.dtype)
        return self.down @ self.up

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: ARG002, B008
    ) -> Float[Array, " out_channels"]:
        if self.is_abstract:
            *leading_dims, out_channels, _rank = self.down.shape
            return dummy_array((*leading_dims, out_channels), self.down.dtype)
        return self.down @ (self.up @ vector)

    def to_uzu(self) -> dict[str, Array]:
        return {
            "down": self.down,
            "up": self.up,
        }

    def from_uzu(self, weights: dict[str, Array]) -> LoraArray:
        return type(self)(
            down=weights["down"],
            up=weights["up"],
        )

    @staticmethod
    def from_rank(
        out_channels: int,
        in_channels: int,
        *,
        rank: int,
        key: PRNGKeyArray,
    ) -> LoraArray:
        key_down, _key_up = jr.split(key)
        return LoraArray(
            down=jr.normal(key_down, (out_channels, rank)) / rank,
            up=jnp.zeros((rank, in_channels)),
        )
