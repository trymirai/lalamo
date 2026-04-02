from __future__ import annotations

import warnings

import equinox as eqx
import jax.core
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.common import ParameterTree

from .base import CompressedArray


class LoraArray(CompressedArray):
    down: Float[Array, "... out_channels rank"]
    up: Float[Array, "... rank in_channels"]
    scale: float = eqx.field(static=True, default=1.0)

    def aval(self) -> jax.core.ShapedArray:
        *batch, out_channels, _ = self.down.shape
        *_, _, in_channels = self.up.shape
        return jax.core.ShapedArray((*batch, out_channels, in_channels), self.down.dtype)

    def dot(self, vector: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        return self.scale * (self.down @ (self.up @ vector))

    def materialise(self) -> Array:
        warnings.warn("LoraArray.materialise() — computing down @ up.", stacklevel=2)
        return self.value

    @property
    def value(self) -> Array:
        return self.scale * (self.down @ self.up)

    def export_weights(self) -> ParameterTree:
        return dict(down_weights=self.down, up_weights=self.up)

    @staticmethod
    def from_rank(
        out_channels: int,
        in_channels: int,
        *,
        rank: int,
        scale: float = 1.0,
        key: PRNGKeyArray,
    ) -> LoraArray:
        key_down, _key_up = jr.split(key)
        return LoraArray(
            down=jr.normal(key_down, (out_channels, rank)) / rank,
            up=jnp.zeros((rank, in_channels)),
            scale=scale,
        )
