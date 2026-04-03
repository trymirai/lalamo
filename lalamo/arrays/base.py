from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange

if TYPE_CHECKING:
    from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.common import ParameterTree


@dataclass(frozen=True)
class NoQuantize:
    pass


@dataclass(frozen=True)
class DeterministicQuantize:
    pass


@dataclass(frozen=True)
class StochasticQuantize:
    key: PRNGKeyArray


Quantize = NoQuantize | DeterministicQuantize | StochasticQuantize


@dataclass(frozen=True)
class ArrayForwardPassConfig:
    quantize: Quantize = DeterministicQuantize()


class CompressedArray(eqx.Module):
    @abc.abstractmethod
    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array: ...  # noqa: B008

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, " out_channels"]:
        return self.materialize(forward_pass_config) @ vector

    @abc.abstractmethod
    def export_weights(self) -> ParameterTree: ...


def unpack_int32(packed: Array, bits: int) -> Array:
    shifts = jnp.arange(0, 32, bits)
    mask = (2**bits) - 1
    unpacked = jnp.bitwise_and(jnp.right_shift(packed[:, :, None], shifts[None, None, :]), mask)
    return rearrange(unpacked, "rows packed_groups packed_vals -> rows (packed_groups packed_vals)")
