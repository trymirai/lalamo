import torch
from jax import numpy as jnp

from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax


def test_dtype_convert_roundtrip() -> None:
    """Test that DTypeConvert correctly converts dtypes between JAX and PyTorch."""
    # Test all supported dtypes: JAX -> PyTorch and back
    test_cases = [
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("bfloat16", torch.bfloat16),
        ("int8", torch.int8),
        ("int16", torch.int16),
        ("int32", torch.int32),
        ("int64", torch.int64),
        ("uint8", torch.uint8),
        ("bool", torch.bool),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]

    for dtype_str, torch_dtype in test_cases:
        jax_dtype = jnp.dtype(dtype_str)

        # Test JAX dtype -> PyTorch
        assert jax_to_torch(jax_dtype) == torch_dtype, f"Failed JAX->Torch for {dtype_str}"

        # Test PyTorch -> JAX
        assert torch_to_jax(torch_dtype) == jax_dtype, f"Failed Torch->JAX for {dtype_str}"

        # Test string -> PyTorch
        assert jax_to_torch(dtype_str) == torch_dtype, f"Failed str->Torch for {dtype_str}"

        # Test string -> JAX
        assert torch_to_jax(dtype_str) == jax_dtype, f"Failed str->JAX for {dtype_str}"
