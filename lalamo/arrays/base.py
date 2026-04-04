from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange

if TYPE_CHECKING:
    from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.common import ParameterTree


class NoQuantize(eqx.Module):
    pass


class DeterministicQuantize(eqx.Module):
    pass


class StochasticQuantize(eqx.Module):
    key: PRNGKeyArray


Quantize = NoQuantize | DeterministicQuantize | StochasticQuantize


class ArrayForwardPassConfig(eqx.Module):
    quantize: Quantize = DeterministicQuantize()


class CompressedArray(eqx.Module):
    @property
    @abc.abstractmethod
    def is_abstract(self) -> bool: ...

    @abc.abstractmethod
    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array: ...  # noqa: B008

    @abc.abstractmethod
    def to_uzu(self) -> dict[str, Array]: ...

    @abc.abstractmethod
    def from_uzu(self, weights: dict[str, Array]) -> CompressedArray: ...

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


def pack_uint_to_uint8(unpacked: Array, bits: int) -> Array:
    if bits == 8:
        return unpacked.astype(jnp.uint8)

    if bits not in {1, 4}:
        raise ValueError(f"Unsupported uint packing width: {bits}")

    if unpacked.ndim == 0:
        raise ValueError("Input array cannot be scalar")

    values_per_byte = 8 // bits
    *_, last_dim = unpacked.shape
    if last_dim % values_per_byte != 0:
        raise ValueError(
            f"Last dimension {last_dim} must be divisible by {values_per_byte} for {bits}-bit packing",
        )

    grouped = rearrange(
        unpacked.astype(jnp.uint8),
        "... (groups packed_values) -> ... groups packed_values",
        packed_values=values_per_byte,
    )
    packed = jnp.zeros(grouped.shape[:-1], dtype=jnp.uint8)
    for shift in range(values_per_byte):
        packed = packed | (grouped[..., shift] << jnp.uint8(shift * bits))
    return packed


def unpack_uint8_to_uint(packed: Array, bits: int) -> Array:
    if bits == 8:
        return packed.astype(jnp.uint8)

    if bits not in {1, 4}:
        raise ValueError(f"Unsupported uint unpacking width: {bits}")

    values_per_byte = 8 // bits
    shifts = jnp.arange(values_per_byte, dtype=jnp.uint8) * jnp.uint8(bits)
    mask = jnp.uint8((1 << bits) - 1)
    unpacked = jnp.bitwise_and(jnp.right_shift(packed[..., None], shifts), mask)
    return rearrange(unpacked, "... groups packed_values -> ... (groups packed_values)")
