from typing import overload

import jax.numpy as jnp
import torch
from jaxtyping import Array

__all__ = ["jax_to_torch", "torch_to_jax"]

_typestr_to_torch_dtype = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
_torch_dtype_to_typestr = {v: k for k, v in _typestr_to_torch_dtype.items()}


def dtype_to_torch(in_type: str | jnp.dtype) -> torch.dtype:
    match in_type:
        case str():
            return _typestr_to_torch_dtype[in_type]
        case _:
            return _typestr_to_torch_dtype[jnp.dtype(in_type).name]


def dtype_to_jax(in_type: str | torch.dtype) -> jnp.dtype:
    match in_type:
        case str():
            return jnp.dtype(in_type)
        case torch.dtype():
            return jnp.dtype(_torch_dtype_to_typestr[in_type])


@torch.no_grad()
def _torch_to_jax_bfloat16(tensor: torch.Tensor) -> Array:
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Trying to convert non-bfloat16 tensor to bfloat16")
    intermediate_tensor = tensor.view(torch.uint16)
    return jnp.array(intermediate_tensor).view("bfloat16")


@overload
def torch_to_jax(value: torch.Tensor) -> Array: ...
@overload
def torch_to_jax(value: str | torch.dtype) -> jnp.dtype: ...
def torch_to_jax(value: torch.Tensor | str | torch.dtype) -> Array | jnp.dtype:  # type: ignore[return-value]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            return _torch_to_jax_bfloat16(value)
        return jnp.array(value.numpy())
    if isinstance(value, str | torch.dtype):
        return dtype_to_jax(value)


@overload
def jax_to_torch(value: Array) -> torch.Tensor: ...
@overload
def jax_to_torch(value: str | jnp.dtype) -> torch.dtype: ...
def jax_to_torch(value: Array | str | jnp.dtype) -> torch.Tensor | torch.dtype:
    if isinstance(value, str | jnp.dtype):
        return dtype_to_torch(value)
    if isinstance(value, Array):
        from torch.utils import dlpack as _dlpack

        if value.dtype == jnp.bfloat16:
            intermediate_array = value.view(jnp.uint16)
            return _dlpack.from_dlpack(intermediate_array).view(torch.bfloat16)
        return _dlpack.from_dlpack(value)
