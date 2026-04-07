import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray


def quantize_to_grid(weights: Float[Array, "..."], bits: int) -> Float[Array, "..."]:
    return jnp.clip(jnp.round(weights), 0, (2**bits) - 1)


def stochastic_quantize_to_grid(weights: Float[Array, "..."], bits: int, key: PRNGKeyArray) -> Float[Array, "..."]:
    qmax = (2**bits) - 1
    floored = jnp.floor(weights)
    frac = weights - floored
    rounded = floored + (jax.random.uniform(key, weights.shape) < frac).astype(weights.dtype)
    return jnp.clip(rounded, 0, qmax)


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
