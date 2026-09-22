import jax.numpy as jnp
from einops import rearrange
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
    _validate_plane(params, columns, groups, bits)
    if layout == Layout.INPUT_OUTPUT:
        return params

    padded_columns = _group_output_columns(columns)
    output_group = _unpack(params, bits, groups)
    group_output = rearrange(output_group, "... output group -> ... group output")
    padding = [(0, 0)] * (group_output.ndim - 1) + [(0, padded_columns - columns)]
    return _pack(jnp.pad(group_output, padding), bits, sharding_config)


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
        _validate_plane(stored, columns, groups, bits)
        return load_as(like, stored)

    padded_columns = _group_output_columns(columns)
    _validate_plane(stored, groups, padded_columns, bits)
    group_output = _unpack(stored, bits, padded_columns)
    output_group = rearrange(group_output[..., :columns], "... group output -> ... output group")
    return load_as(like, _pack(output_group, bits, sharding_config))


def _group_output_columns(columns: int) -> int:
    columns_per_load = 4
    return columns + (-columns % columns_per_load)


def _unpack(array: Array, bits: int, width: int) -> Array:
    if bits in (4, 8):
        return unpack_uint8_to_uint(array, bits=bits, unpacked_last_axis_dim=width)
    return array


def _pack(array: Array, bits: int, sharding_config: ShardingConfig) -> Array:
    if bits in (4, 8):
        return pack_uint_to_uint8(array, bits, sharding_config=sharding_config)
    return array


def _validate_plane(array: Array, rows: int, columns: int, bits: int) -> None:
    if rows <= 0 or columns <= 0:
        raise ValueError(f"quantization parameter dimensions must be positive: rows={rows}, columns={columns}")
    if bits in (4, 8):
        columns = packed_last_axis_dim(columns, bits)
    elif bits not in (16, 32):
        raise ValueError(f"unsupported quantization parameter width: {bits}")
    if tuple(array.shape[-2:]) != (rows, columns):
        raise ValueError(f"quantization parameter shape {array.shape} does not end with {(rows, columns)}")
