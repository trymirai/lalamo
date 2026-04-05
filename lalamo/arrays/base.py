from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange

from lalamo.common import stringify_path

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


def _grouped_init_stats(
    raw: Float[Array, "... out_channels in_channels"], group_size: int, bits: int
) -> tuple[Float[Array, "... out_channels groups"], Float[Array, "... out_channels groups"]]:
    *leading_dims, out_channels, in_channels = raw.shape
    if in_channels % group_size != 0:
        raise ValueError(f"in_channels ({in_channels}) must be divisible by group_size ({group_size})")
    grouped = raw.reshape((*leading_dims, out_channels, in_channels // group_size, group_size))
    group_mins = jnp.min(grouped, axis=-1)
    group_maxs = jnp.max(grouped, axis=-1)
    quant_levels = (2**bits) - 1
    scales = jnp.maximum((group_maxs - group_mins) / quant_levels, jnp.finfo(raw.dtype).eps)
    return group_mins, scales


class CompressedArray(eqx.Module):
    @property
    @abc.abstractmethod
    def is_abstract(self) -> bool: ...

    @abc.abstractmethod
    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array: ...  # noqa: B008

    def to_uzu(self) -> dict[str, Array]:
        flat_with_path, _ = jax.tree_util.tree_flatten_with_path(
            self,
            is_leaf=lambda x: isinstance(x, CompressedArray) and x is not self,
        )
        result: dict[str, Array] = {}
        for path, leaf in flat_with_path:
            key = stringify_path(path)
            if isinstance(leaf, CompressedArray):
                for sub_key, array in leaf.to_uzu().items():
                    result[f"{key}.{sub_key}"] = array
            else:
                result[key] = leaf
        return result

    def from_uzu(self, weights: dict[str, Array]) -> CompressedArray:
        def restore(path: tuple[object, ...], leaf: object) -> object:
            key = stringify_path(path)
            if isinstance(leaf, CompressedArray):
                relevant = {k.removeprefix(f"{key}."): weights[k] for k in weights if k.startswith(f"{key}.")}
                return leaf.from_uzu(relevant)
            if key in weights:
                return weights[key]
            return leaf

        return jax.tree_util.tree_map_with_path(
            restore,
            self,
            is_leaf=lambda x: isinstance(x, CompressedArray) and x is not self,
        )

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
