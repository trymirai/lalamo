import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike

from lalamo.utils.dummy_array import preserve_first_input_sharding, supports_dummy_arrays

__all__ = [
    "pack_uint_to_uint8",
    "unpack_uint8_to_uint",
]


def _values_per_byte(bits: int) -> int:
    if bits not in {1, 4}:
        raise ValueError(f"Unsupported uint packing width: {bits}")
    return 8 // bits


def packed_last_axis_dim(last_dim: int, bits: int) -> int:
    if bits == 8:
        return last_dim
    values_per_byte = _values_per_byte(bits)
    if last_dim % values_per_byte != 0:
        raise ValueError(f"Last dimension {last_dim} must be divisible by {values_per_byte} for {bits}-bit packing")
    return last_dim // values_per_byte


def _packed_shape(shape: tuple[int, ...], bits: int) -> tuple[int, ...]:
    if not shape:
        raise ValueError("Input array cannot be scalar")
    *leading_dims, last_dim = shape
    return (*leading_dims, packed_last_axis_dim(last_dim, bits))


@supports_dummy_arrays()
def pack_uint_to_uint8(unpacked: Array, bits: int) -> Array:
    if bits == 8:
        return unpacked.astype(jnp.uint8)

    values_per_byte = _values_per_byte(bits)
    _packed_shape(unpacked.shape, bits)

    grouped = rearrange(
        unpacked.astype(jnp.uint8),
        "... (groups packed_values) -> ... groups packed_values",
        packed_values=values_per_byte,
    )
    shifts = jnp.arange(values_per_byte, dtype=jnp.uint8) * jnp.uint8(bits)
    return jnp.sum(grouped << shifts, axis=-1, dtype=jnp.uint8)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def unpack_uint8_to_uint(
    packed: Array,
    bits: int,
    *,
    dtype: DTypeLike = jnp.uint8,
) -> Array:
    if bits == 8:
        return packed.astype(dtype)

    values_per_byte = _values_per_byte(bits)
    shifts = jnp.arange(values_per_byte, dtype=jnp.uint8) * jnp.uint8(bits)
    mask = jnp.uint8((1 << bits) - 1)
    unpacked = jnp.bitwise_and(jnp.right_shift(packed[..., None], shifts), mask)
    return rearrange(unpacked, "... groups packed_values -> ... (groups packed_values)").astype(dtype)
