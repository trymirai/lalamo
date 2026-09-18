import jax.numpy as jnp
from jaxtyping import Array

from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import QuantParamsLayout

from .utils.packing import pack_uint_to_uint8, unpack_uint8_to_uint


def for_export(
    params: Array,
    *,
    shape: tuple[int, int],
    layout: QuantParamsLayout,
    bits: int,
    sharding_config: ShardingConfig,
) -> Array:
    return _convert(
        params,
        shape,
        bits,
        QuantParamsLayout.OUTPUT_GROUP,
        layout,
        sharding_config,
    )


def from_export(
    stored: Array,
    *,
    like: Array,
    shape: tuple[int, int],
    layout: QuantParamsLayout,
    bits: int,
    sharding_config: ShardingConfig,
) -> Array:
    return load_as(
        like,
        _convert(
            stored,
            shape,
            bits,
            layout,
            QuantParamsLayout.OUTPUT_GROUP,
            sharding_config,
        ),
    )


def _convert(
    array: Array,
    shape: tuple[int, int],
    bits: int,
    source_layout: QuantParamsLayout,
    target_layout: QuantParamsLayout,
    sharding_config: ShardingConfig,
) -> Array:
    columns, groups = shape
    if source_layout == target_layout:
        return array
    if columns <= 0 or groups <= 0:
        raise ValueError(f"quantization parameter dimensions must be positive: columns={columns}, groups={groups}")
    if bits not in (4, 8, 16, 32):
        raise ValueError(f"unsupported quantization parameter width: {bits}")

    stride = (columns + 3) // 4 * 4
    source_last_dim = groups if source_layout == QuantParamsLayout.OUTPUT_GROUP else stride
    if bits in (4, 8):
        logical = unpack_uint8_to_uint(array, bits=bits, unpacked_last_axis_dim=source_last_dim)
        source_last_dim = (source_last_dim * bits + 7) // 8
    else:
        logical = array
    expected_shape = (
        *array.shape[:-2],
        columns if source_layout == QuantParamsLayout.OUTPUT_GROUP else groups,
        source_last_dim,
    )
    if tuple(array.shape[-2:]) != expected_shape[-2:]:
        raise ValueError(f"quantization parameter shape {array.shape} does not match {expected_shape}")

    if source_layout == QuantParamsLayout.OUTPUT_GROUP:
        logical = jnp.swapaxes(logical, -2, -1)
        stride = (columns + 3) // 4 * 4
        logical = jnp.pad(logical, [(0, 0)] * (logical.ndim - 1) + [(0, stride - columns)])
    else:
        logical = jnp.swapaxes(logical[..., :columns], -2, -1)

    if bits in (4, 8):
        return pack_uint_to_uint8(logical, bits, sharding_config=sharding_config)
    return logical
