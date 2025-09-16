from collections.abc import Callable

import jax
from jax import vmap
from jaxtyping import Array, Float

__all__ = [
    "apply_soft_capping",
    "vmap_twice",
]


def vmap_twice[F: Callable](
    func: F,
) -> F:
    return vmap(vmap(func, in_axes=0), in_axes=0)


def apply_soft_capping(
    values: Float[Array, "*"],
    soft_cap: float,
) -> Float[Array, "dst_tokens src_tokens"]:
    return jax.nn.tanh(values / soft_cap) * soft_cap
