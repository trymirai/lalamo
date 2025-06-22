import einops
import jax.numpy as jnp
import torch
from jaxtyping import Array

__all__ = [
    "jax_int4_to_packed_uint8",
    "jax_to_torch",
    "torch_to_jax",
]


@torch.no_grad()
def _torch_to_jax_bfloat16(tensor: torch.Tensor) -> Array:
    # Credit: https://github.com/jax-ml/ml_dtypes/issues/81#issuecomment-2399636232
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Trying to convert non-bfloat16 tensor to bfloat16")
    intermediate_tensor = tensor.view(torch.uint16)
    return jnp.array(intermediate_tensor).view("bfloat16")


def torch_to_jax(array: torch.Tensor) -> Array:
    array = array.detach().cpu()
    if array.dtype == torch.bfloat16:
        return _torch_to_jax_bfloat16(array)
    return jnp.array(array.numpy())


def jax_to_torch(array: Array) -> torch.Tensor:
    if array.dtype == jnp.bfloat16:
        intermediate_array = array.view(jnp.uint16)
        return torch.tensor(intermediate_array).view(torch.bfloat16)
    return torch.tensor(array)


def jax_int4_to_packed_uint8(array: Array) -> Array:
    if array.dtype != jnp.int4:
        raise ValueError(f"Input array must have dtype jnp.int4, but got {array.dtype}")

    if not array.shape:
        raise ValueError("Input array cannot be a scalar and must have at least one dimension.")

    *_, last_dim = array.shape
    if last_dim % 2 != 0:
        raise ValueError(f"The last dimension of the input array must be even, but got shape {array.shape}")

    # Convert int4 values to their unsigned 4-bit representation by taking the bit pattern.
    # jnp.int4 values are in the range [-8, 7]. `astype(jnp.int8)` preserves these values.
    # `& 0x0F` then gives us the 4-bit two's-complement bit pattern as an integer.
    unsigned_4bit = array.astype(jnp.int8) & 0x0F

    # Split the array into low and high nibbles. The first element of each pair is the low nibble,
    # and the second is the high nibble.
    low_nibbles, high_nibbles = einops.rearrange(
        unsigned_4bit,
        "... (dim_half two) -> two ... dim_half",
        two=2,
    )

    # Pack the two 4-bit values into a single uint8.
    # high_nibbles are shifted to the left by 4 bits to become the most significant bits.
    packed = (high_nibbles << 4) | low_nibbles

    return packed.astype(jnp.uint8)
