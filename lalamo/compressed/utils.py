from typing import NamedTuple

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float

__all__ = [
    "MinMax",
    "group_by_last_axis",
]


class MinMax(NamedTuple):
    min: Float[Array, "... out_channels in_channels"]
    max: Float[Array, "... out_channels in_channels"]


def group_by_last_axis(
    weights: Float[Array, "... out_channels in_channels"],
    *,
    group_size: int,
) -> Float[Array, "... out_channels groups group_channels"]:
    return rearrange(
        weights,
        "... out_channels (groups group_size) -> ... out_channels groups group_size",
        group_size=group_size,
    )


def min_max_within_groups(weights: Float[Array, "... out_channels groups group_channels"]) -> MinMax:
    return MinMax(
        min=jnp.min(weights, axis=-1),
        max=jnp.max(weights, axis=-1),
    )


def round_to_integers(
    values: Float[Array, "..."],
    *,
    num_bits: int = 8,
) -> Float[Array, "..."]:
    min_value = -(2 ** (num_bits - 1))
    max_value = 2 ** (num_bits - 1) - 1
    return jnp.clip(jnp.round(values), min_value, max_value)
