from typing import Any

import jax.numpy as jnp

__all__ = ["DEFAULT_PRECISION", "DType"]


type DType = jnp.dtype | Any  # This is a hack to make the typechecker happy with internals of JAX.


DEFAULT_PRECISION: DType = jnp.float32
