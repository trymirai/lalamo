import einops
import jax.numpy as jnp
from jaxtyping import Array

__all__ = ["jax_uint4_to_packed_uint8"]


def jax_uint4_to_packed_uint8(array: Array) -> Array:
    if array.dtype != jnp.uint4:
        raise ValueError(f"Input array must have dtype jnp.uint4, but got {array.dtype}")

    if not array.shape:
        raise ValueError("Input array cannot be a scalar and must have at least one dimension.")

    *_, last_dim = array.shape
    if last_dim % 2 != 0:
        raise ValueError(f"The last dimension of the input array must be even, but got shape {array.shape}")

    low_nibbles, high_nibbles = einops.rearrange(
        array.astype(jnp.uint8),
        "... (dim_half two) -> two ... dim_half",
        two=2,
    )

    packed = (high_nibbles << 4) | low_nibbles

    return packed.astype(jnp.uint8)
