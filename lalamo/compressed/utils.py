from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float

from lalamo.utils.dummy_array import dummy_array

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
    min: Float[Array, "... out_channels in_channels"]
    max: Float[Array, "... out_channels in_channels"]


def grouped_last_axis_shape(shape: tuple[int, ...], *, group_size: int) -> tuple[int, ...]:
    *leading_dims, last_dim = shape
    if last_dim % group_size != 0:
        raise ValueError(f"Last dimension {last_dim} must be divisible by group size {group_size}")
    return (*leading_dims, last_dim // group_size)


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


def expand_last_axis_groups[ArrayT: Array | ShapeDtypeStruct](
    grouped: Float[ArrayT, "... groups"],
    *,
    group_size: int,
) -> Float[Array, "..."]:
    if isinstance(grouped, ShapeDtypeStruct):
        *leading_dims, groups = grouped.shape
        return dummy_array((*leading_dims, groups * group_size), grouped.dtype, grouped.sharding)
    assert isinstance(grouped, jax.Array)
    return jnp.repeat(grouped, group_size, axis=-1)


def unsigned_qmax(bits: int) -> int:
    return (2**bits) - 1


def min_max_within_groups(weights: Float[Array, "... out_channels groups group_channels"]) -> MinMax:
    return MinMax(
        min=jnp.min(weights, axis=-1),
        max=jnp.max(weights, axis=-1),
    )


def scale_from_min_max(min_max: MinMax, *, bits: int, dtype: DTypeLike) -> Float[Array, "..."]:
    finfo = jnp.finfo(dtype)
    scale_range = min_max.max / unsigned_qmax(bits) - min_max.min / unsigned_qmax(bits)
    scales = jnp.maximum(scale_range, finfo.eps)
    return jnp.nan_to_num(scales, nan=finfo.eps, posinf=finfo.max, neginf=finfo.eps)
