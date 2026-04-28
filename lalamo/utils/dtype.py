import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

__all__ = [
    "cast_if_float",
    "dtype_to_str",
    "str_to_dtype",
]


_ALL_DTYPES = (
    jnp.int4,
    jnp.int8,
    jnp.int16,
    jnp.int32,
    jnp.int64,
    jnp.float4_e2m1fn,
    jnp.float8_e3m4,
    jnp.float8_e4m3fn,
    jnp.float8_e4m3fnuz,
    jnp.float8_e4m3b11fnuz,
    jnp.float8_e5m2,
    jnp.float8_e5m2fnuz,
    jnp.float8_e8m0fnu,
    jnp.bfloat16,
    jnp.float16,
    jnp.float32,
    jnp.float64,
    jnp.complex64,
    jnp.complex128,
)


def dtype_to_str(dtype: DTypeLike) -> str:
    if dtype == jnp.bfloat16:
        return "bfloat16"
    try:
        return str(dtype.dtype)  # type: ignore[missing-attribute]
    except AttributeError:
        return str(dtype)


def str_to_dtype(dtype_str: str) -> jnp.dtype:
    return {str(dtype): dtype for dtype in _ALL_DTYPES}[dtype_str]


def cast_if_float(array: Array, dtype: DTypeLike) -> Array:
    return array.astype(dtype) if jnp.issubdtype(array.dtype, jnp.floating) else array
