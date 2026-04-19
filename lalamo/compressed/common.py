from collections.abc import Sequence

from einops import rearrange, repeat
from jaxtyping import Array, Float

from lalamo.module import ShardingAxis
from lalamo.weight_matrix import Layout

__all__ = [
    "expand_group_parameter",
    "group_by_input_axis",
    "group_parameter_partition",
    "lookup_group_parameter",
    "lookup_output",
    "matmul_with_layout",
    "store_weights",
    "weight_matrix_partition",
]


def store_weights(
    weights: Float[Array, "... out_channels in_channels"],
    layout: Layout,
) -> Float[Array, "..."]:
    if layout == Layout.INPUT_OUTPUT:
        return weights.swapaxes(-1, -2)
    return weights


def group_by_input_axis(
    weights: Float[Array, "..."],
    *,
    layout: Layout,
    group_size: int,
) -> Float[Array, "..."]:
    if layout == Layout.OUTPUT_INPUT:
        return rearrange(
            weights,
            "... out_channels (groups group_size) -> ... out_channels groups group_size",
            group_size=group_size,
        )
    return rearrange(
        weights,
        "... (groups group_size) out_channels -> ... groups out_channels group_size",
        group_size=group_size,
    )


def expand_group_parameter(
    grouped: Float[Array, "..."],
    *,
    layout: Layout,
    group_size: int,
) -> Float[Array, "..."]:
    if layout == Layout.OUTPUT_INPUT:
        return repeat(
            grouped,
            "... out_channels groups -> ... out_channels (groups group_size)",
            group_size=group_size,
        )
    return repeat(
        grouped,
        "... groups out_channels -> ... (groups group_size) out_channels",
        group_size=group_size,
    )


def lookup_group_parameter(
    grouped: Float[Array, "..."],
    *,
    layout: Layout,
    index: int | Array,
) -> Float[Array, "..."]:
    if layout == Layout.OUTPUT_INPUT:
        return grouped[index, :]
    return grouped[:, index]


def lookup_output(
    values: Float[Array, "..."],
    *,
    layout: Layout,
    index: int | Array,
) -> Float[Array, "..."]:
    if layout == Layout.OUTPUT_INPUT:
        return values[index, :]
    return values[:, index]


def matmul_with_layout(
    matrix: Float[Array, "..."],
    vector: Float[Array, " in_channels"],
    *,
    layout: Layout,
) -> Float[Array, "..."]:
    if layout == Layout.INPUT_OUTPUT:
        return vector @ matrix
    return matrix @ vector


def _leading_partition(leading_dims: Sequence[int]) -> tuple[ShardingAxis, ...]:
    return (ShardingAxis.EXPERT,) * len(leading_dims)


def weight_matrix_partition(
    layout: Layout,
    leading_dims: Sequence[int],
) -> tuple[ShardingAxis | None, ...]:
    trailing_partition: tuple[ShardingAxis | None, ShardingAxis | None]
    if layout == Layout.INPUT_OUTPUT:
        trailing_partition = (None, ShardingAxis.TENSOR)
    else:
        trailing_partition = (ShardingAxis.TENSOR, None)
    return _leading_partition(leading_dims) + trailing_partition


def group_parameter_partition(
    layout: Layout,
    leading_dims: Sequence[int],
) -> tuple[ShardingAxis | None, ...]:
    trailing_partition: tuple[ShardingAxis | None, ShardingAxis | None]
    if layout == Layout.INPUT_OUTPUT:
        trailing_partition = (None, ShardingAxis.TENSOR)
    else:
        trailing_partition = (ShardingAxis.TENSOR, None)
    return _leading_partition(leading_dims) + trailing_partition
