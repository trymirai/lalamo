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
    "packed_uint4_group_dot",
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


def packed_uint4_group_dot(
    packed_weights: Array,
    vector: Float[Array, " channels"],
    group_size: int,
) -> tuple[Float[Array, "rows groups"], Float[Array, " groups"]]:
    rows, packed_cols = packed_weights.shape
    groups = vector.shape[0] // group_size
    packed_group_size = group_size // 2
    assert group_size % 2 == 0
    assert packed_cols == groups * packed_group_size

    grouped = packed_weights.reshape(rows, groups, packed_group_size)
    lo = jnp.bitwise_and(grouped, jnp.uint8(0x0F)).astype(jnp.float32)
    hi = jnp.bitwise_and(jnp.right_shift(grouped, jnp.uint8(4)), jnp.uint8(0x0F)).astype(jnp.float32)

    vector_groups = vector.reshape(groups, group_size).astype(jnp.float32)
    even_values = vector_groups[:, 0::2]
    odd_values = vector_groups[:, 1::2]
    int_dot = jnp.sum(lo * even_values[None, :, :] + hi * odd_values[None, :, :], axis=-1)
    vector_sums = jnp.sum(vector_groups, axis=-1)
    return int_dot, vector_sums


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
