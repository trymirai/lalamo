from __future__ import annotations

import abc
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float

from lalamo.common import ParameterTree


@dataclass(frozen=True)
class ArrayForwardPassConfig:
    quantize: bool = False


class CompressedArray(eqx.Module):
    @abc.abstractmethod
    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array: ...

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),
    ) -> Float[Array, " out_channels"]:
        return self.materialize(forward_pass_config) @ vector

    @abc.abstractmethod
    def export_weights(self) -> ParameterTree: ...


def unpack_int32(packed: Array, bits: int) -> Array:
    shifts = jnp.arange(0, 32, bits)
    mask = (2**bits) - 1
    unpacked = jnp.bitwise_and(jnp.right_shift(packed[:, :, None], shifts[None, None, :]), mask)
    return rearrange(unpacked, "rows packed_groups packed_vals -> rows (packed_groups packed_vals)")
