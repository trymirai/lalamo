import math

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike

from lalamo.utils.dummy_array import preserve_first_input_sharding, supports_dummy_arrays
from lalamo.utils.sharding import ShardingConfig, sharding_of, with_sharding

__all__ = [
    "pack_uint_to_uint8",
    "unpack_uint8_to_uint",
]


def _check_bits(bits: int) -> None:
    if bits not in {1, 2, 3, 4, 6, 8}:
        raise ValueError(f"Unsupported uint packing width: {bits}")


def _values_per_byte(bits: int) -> int:
    if 8 % bits != 0:
        raise ValueError(f"{bits}-bit values do not evenly fit into a byte")
    return 8 // bits


def packed_last_axis_dim(last_dim: int, bits: int) -> int:
    _check_bits(bits)
    if bits == 8:
        return last_dim
    return math.ceil(last_dim * bits / 8)


def _packed_shape(shape: tuple[int, ...], bits: int) -> tuple[int, ...]:
    if not shape:
        raise ValueError("Input array cannot be scalar")
    *leading_dims, last_dim = shape
    return (*leading_dims, packed_last_axis_dim(last_dim, bits))


@supports_dummy_arrays()
def pack_uint_to_uint8(unpacked: Array, bits: int, *, sharding_config: ShardingConfig) -> Array:
    _check_bits(bits)
    unpacked_sharding = sharding_of(unpacked)
    scratch_sharding = sharding_config.make_sharding((None,) * unpacked.ndim)
    if bits == 8:
        return with_sharding(unpacked.astype(jnp.uint8), unpacked_sharding)

    packed_shape = _packed_shape(unpacked.shape, bits)
    *_, packed_last_dim = packed_shape
    *_, unpacked_last_dim = unpacked.shape
    *leading_specs, _packed_axis_spec = unpacked_sharding.spec
    packed_sharding = sharding_config.make_sharding((*leading_specs, None))
    if 8 % bits == 0 and unpacked_last_dim % _values_per_byte(bits) == 0:
        values_per_byte = _values_per_byte(bits)
        grouped = rearrange(
            unpacked.astype(jnp.uint8),
            "... (groups packed_values) -> ... groups packed_values",
            packed_values=values_per_byte,
        )
        shifts = jnp.arange(values_per_byte, dtype=jnp.uint8) * jnp.uint8(bits)
        packed = jnp.sum(grouped << shifts, axis=-1, dtype=jnp.uint8)
        return with_sharding(packed, packed_sharding)

    values = unpacked.astype(jnp.uint32) & jnp.uint32((1 << bits) - 1)
    bit_offsets = jnp.arange(unpacked_last_dim, dtype=jnp.uint32) * jnp.uint32(bits)
    byte_indices = bit_offsets // jnp.uint32(8)
    bit_indices = bit_offsets % jnp.uint32(8)

    low_parts = (values << bit_indices) & jnp.uint32(0xFF)
    packed = with_sharding(jnp.zeros(packed_shape, dtype=jnp.uint32), scratch_sharding)
    low_parts = with_sharding(low_parts, scratch_sharding)
    packed = packed.at[..., byte_indices].add(low_parts)

    crosses_byte = bit_indices + jnp.uint32(bits) > jnp.uint32(8)
    high_parts = jnp.where(crosses_byte, values >> (jnp.uint32(8) - bit_indices), jnp.uint32(0))
    high_parts = with_sharding(high_parts, scratch_sharding)
    next_byte_indices = jnp.minimum(byte_indices + jnp.uint32(1), jnp.uint32(packed_last_dim - 1))
    packed = packed.at[..., next_byte_indices].add(high_parts)

    return with_sharding(packed.astype(jnp.uint8), packed_sharding)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def unpack_uint8_to_uint(
    packed: Array,
    bits: int,
    *,
    dtype: DTypeLike = jnp.uint8,
    unpacked_last_axis_dim: int | None = None,
) -> Array:
    _check_bits(bits)
    if bits == 8:
        if unpacked_last_axis_dim is None:
            return packed.astype(dtype)
        return packed[..., :unpacked_last_axis_dim].astype(dtype)

    *_, packed_last_dim = packed.shape
    if unpacked_last_axis_dim is None:
        unpacked_last_axis_dim = packed_last_dim * 8 // bits
    if 8 % bits == 0:
        values_per_byte = _values_per_byte(bits)
        shifts = jnp.arange(values_per_byte, dtype=jnp.uint8) * jnp.uint8(bits)
        mask = jnp.uint8((1 << bits) - 1)
        unpacked = jnp.bitwise_and(jnp.right_shift(packed[..., None], shifts), mask)
        unpacked = rearrange(unpacked, "... groups packed_values -> ... (groups packed_values)")
        return unpacked[..., :unpacked_last_axis_dim].astype(dtype)

    values = packed.astype(jnp.uint32)
    bit_offsets = jnp.arange(unpacked_last_axis_dim, dtype=jnp.uint32) * jnp.uint32(bits)
    byte_indices = bit_offsets // jnp.uint32(8)
    bit_indices = bit_offsets % jnp.uint32(8)
    next_byte_indices = jnp.minimum(byte_indices + jnp.uint32(1), jnp.uint32(packed_last_dim - 1))

    low_parts = values[..., byte_indices] >> bit_indices
    crosses_byte = bit_indices + jnp.uint32(bits) > jnp.uint32(8)
    high_parts = jnp.where(
        crosses_byte,
        values[..., next_byte_indices] << (jnp.uint32(8) - bit_indices),
        jnp.uint32(0),
    )
    unpacked = (low_parts | high_parts) & jnp.uint32((1 << bits) - 1)
    return unpacked.astype(dtype)
