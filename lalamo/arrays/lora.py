from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.common import ParameterTree

from .base import ArrayForwardPassConfig, CompressedArray


class LoraArray(CompressedArray):
    down: Float[Array, "... out_channels rank"]
    up: Float[Array, "... rank in_channels"]

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:
        return self.down @ self.up

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),
    ) -> Float[Array, " out_channels"]:
        return self.down @ (self.up @ vector)

    def export_weights(self) -> ParameterTree:
        return dict(down_weights=self.down, up_weights=self.up)

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
