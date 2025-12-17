import jax.numpy as jnp
import mlx.core as mx  # type: ignore
from jaxtyping import Array

__all__ = ["jax_to_mlx", "mlx_to_jax"]


def mlx_to_jax(a: mx.array) -> Array:
    if a.dtype == mx.bfloat16:
        return jnp.asarray(a.view(mx.uint16)).view(jnp.bfloat16)

    return jnp.asarray(a)


def jax_to_mlx(a: Array) -> mx.array:
    if a.dtype == jnp.bfloat16:
        return mx.array(a.view(jnp.uint16)).view(mx.bfloat16)  # type: ignore

    return mx.array(a)  # type: ignore
