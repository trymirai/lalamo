from typing import NamedTuple

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float

from lalamo.utils.dummy_array import preserve_first_input_sharding, supports_dummy_arrays

__all__ = [
    "MinMax",
    "expand_last_axis_groups",
    "group_by_last_axis",
    "grouped_last_axis_shape",
    "min_max_within_groups",
    "scale_from_min_max",
    "unsigned_qmax",
]


class MinMax(NamedTuple):
    min: Float[Array, "... rows groups"]
    max: Float[Array, "... rows groups"]


def grouped_last_axis_shape(shape: tuple[int, ...], *, group_size: int) -> tuple[int, ...]:
    *leading_dims, last_dim = shape
    if last_dim % group_size != 0:
        raise ValueError(f"Last dimension {last_dim} must be divisible by group size {group_size}")
    return (*leading_dims, last_dim // group_size)


def group_by_last_axis(
    weights: Float[Array, "... rows cols"],
    *,
    group_size: int,
) -> Float[Array, "... rows groups group_channels"]:
    grouped_last_axis_shape(weights.shape, group_size=group_size)
    return rearrange(
        weights,
        "... out_channels (groups group_size) -> ... out_channels groups group_size",
        group_size=group_size,
    )


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def expand_last_axis_groups(
    grouped: Float[Array, "... groups"],
    *,
    group_size: int,
) -> Float[Array, "..."]:
    expanded = jnp.broadcast_to(grouped[..., None], (*grouped.shape, group_size))
    return rearrange(expanded, "... groups group_size -> ... (groups group_size)")


def unsigned_qmax(bits: int) -> int:
    return (2**bits) - 1


def min_max_within_groups(weights: Float[Array, "... rows groups group_channels"]) -> MinMax:
    return MinMax(
        min=jnp.min(weights, axis=-1),
        max=jnp.max(weights, axis=-1),
    )


def scale_from_min_max(min_max: MinMax, *, bits: int, dtype: DTypeLike) -> Float[Array, "..."]:
    finfo = jnp.finfo(dtype)
    scale_range = min_max.max / unsigned_qmax(bits) - min_max.min / unsigned_qmax(bits)
    scales = jnp.maximum(scale_range, finfo.eps)
    return jnp.nan_to_num(scales, nan=finfo.eps, posinf=finfo.max, neginf=finfo.eps)
