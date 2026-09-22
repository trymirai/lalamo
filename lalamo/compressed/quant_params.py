import jax.numpy as jnp
from jaxtyping import Array

from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import Layout

from .utils.packing import pack_uint_to_uint8, packed_last_axis_dim, unpack_uint8_to_uint


def for_export(
    params: Array,
    *,
    shape: tuple[int, int],
    layout: Layout,
    bits: int,
    sharding_config: ShardingConfig,
) -> Array:
    columns, groups = shape
    _check_shape(params, (columns, _stored_width(groups, bits)))
    if layout == Layout.INPUT_OUTPUT:
        return params

    stride = (columns + 3) // 4 * 4
    logical = _unpack(params, bits, groups)
    logical = jnp.pad(jnp.swapaxes(logical, -2, -1), [(0, 0)] * (logical.ndim - 1) + [(0, stride - columns)])
    return _pack(logical, bits, sharding_config)


def from_export(
    stored: Array,
    *,
    like: Array,
    shape: tuple[int, int],
    layout: Layout,
    bits: int,
    sharding_config: ShardingConfig,
) -> Array:
    columns, groups = shape
    if layout == Layout.INPUT_OUTPUT:
        _check_shape(stored, (columns, _stored_width(groups, bits)))
        return load_as(like, stored)

    stride = (columns + 3) // 4 * 4
    _check_shape(stored, (groups, _stored_width(stride, bits)))
    logical = _unpack(stored, bits, stride)
    logical = jnp.swapaxes(logical[..., :columns], -2, -1)
    return load_as(like, _pack(logical, bits, sharding_config))


def _unpack(array: Array, bits: int, width: int) -> Array:
    if bits in (4, 8):
        return unpack_uint8_to_uint(array, bits=bits, unpacked_last_axis_dim=width)
    return array


def _pack(array: Array, bits: int, sharding_config: ShardingConfig) -> Array:
    if bits in (4, 8):
        return pack_uint_to_uint8(array, bits, sharding_config=sharding_config)
    return array


def _stored_width(width: int, bits: int) -> int:
    if bits in (4, 8):
        return packed_last_axis_dim(width, bits)
    return width


def _check_shape(array: Array, expected_tail: tuple[int, int]) -> None:
    if tuple(array.shape[-2:]) != expected_tail:
        raise ValueError(f"quantization parameter shape {array.shape} does not match *{expected_tail}")
