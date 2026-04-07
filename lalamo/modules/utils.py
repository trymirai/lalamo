from collections.abc import Callable
from functools import partial
from typing import Any

import jax
from jax import vmap
from jaxtyping import Array, Float, PRNGKeyArray

__all__ = [
    "apply_soft_capping",
    "vmap_twice",
    "vmap_with_key",
]


def vmap_twice[F: Callable](
    func: F,
) -> F:
    return vmap(vmap(func, in_axes=0), in_axes=0)


def vmap_with_key(
    fn: Callable[..., Any],
    inputs: Array,
    *,
    key: PRNGKeyArray | None,
) -> Any:
    if key is None:
        return vmap(partial(fn, key=None))(inputs)
    keys = jax.random.split(key, inputs.shape[0])
    return vmap(lambda x, k: fn(x, key=k))(inputs, keys)


def apply_soft_capping(
    values: Float[Array, "*"],
    soft_cap: float,
) -> Float[Array, "dst_tokens src_tokens"]:
    return jax.nn.tanh(values / soft_cap) * soft_cap
