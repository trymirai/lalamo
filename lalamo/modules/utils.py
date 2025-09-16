from collections.abc import Callable

import jax
from jax import vmap
from jaxtyping import Array, Float

__all__ = [
    "apply_soft_capping",
    "vmap_twice",
]


def vmap_twice(
    func: Callable[[Float[Array, "*axes"]], Float[Array, "*axes"]],
) -> Callable[[Float[Array, "batch tokens *axes"]], Float[Array, "batch tokens *axes"]]:
    return vmap(vmap(func, in_axes=0), in_axes=0)


def apply_soft_capping(
    values: Float[Array, "*"],
    soft_cap: float,
) -> Float[Array, "dst_tokens src_tokens"]:
    return jax.nn.tanh(values / soft_cap) * soft_cap
