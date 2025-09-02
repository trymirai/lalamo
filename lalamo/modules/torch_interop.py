import jax.numpy as jnp
import torch
from jaxtyping import Array

__all__ = ["jax_to_torch", "torch_to_jax"]


@torch.no_grad()
def _torch_to_jax_bfloat16(tensor: torch.Tensor) -> Array:
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
    from torch.utils import dlpack as _dlpack

    if array.dtype == jnp.bfloat16:
        intermediate_array = array.view(jnp.uint16)
        return _dlpack.from_dlpack(intermediate_array).view(torch.bfloat16)
    return _dlpack.from_dlpack(array)
