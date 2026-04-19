from collections.abc import Callable
from typing import Any

import jax
from jax import vmap
from jaxtyping import Array, Float, Key

__all__ = [
    "apply_soft_capping",
    "vmap_twice",
    "vmap_twice_with_dequant_key",
    "vmap_with_dequant_key",
]


def vmap_twice[F: Callable](func: F) -> F:
    return vmap(vmap(func, in_axes=0), in_axes=0)


def vmap_with_dequant_key(
    fn: Callable[..., Any],
    inputs: Array,
    *,
    dequant_key: Key[Array, ""],
) -> Any:  # noqa: ANN401
    return vmap(lambda x: fn(x, dequant_key=dequant_key))(inputs)


def vmap_twice_with_dequant_key(
    fn: Callable[..., Any],
    inputs: Array,
    *,
    dequant_key: Key[Array, ""],
) -> Any:  # noqa: ANN401
    return vmap_with_dequant_key(
        lambda batch, *, dequant_key: vmap_with_dequant_key(
            fn,
            batch,
            dequant_key=dequant_key,
        ),
        inputs,
        dequant_key=dequant_key,
    )


def apply_soft_capping(
    values: Float[Array, "*"],
    soft_cap: float,
) -> Float[Array, "dst_tokens src_tokens"]:
    return jax.nn.tanh(values / soft_cap) * soft_cap
