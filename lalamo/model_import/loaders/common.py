from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
import jax
import jax.sharding as shd
import jax.tree_util as jtu

from lalamo.common import _is_leaf_array
from lalamo.modules.common import (
    FieldMetadataInfo,
    ShardingConfig,
    _field_info_at_path,
    get_current_sharding_config,
)

__all__ = [
    "load_parameters",
]


SelectedTree = TypeVar("SelectedTree")


def _abstract_signature(tree: object) -> object:
    return jax.tree.map(
        lambda leaf: (
            None
            if leaf is None
            else jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
            if _is_leaf_array(leaf)
            else type(leaf)
        ),
        tree,
        is_leaf=lambda leaf: leaf is None or _is_leaf_array(leaf),
    )


def _apply_parameter_sharding(
    array: jax.Array,
    field_info: FieldMetadataInfo,
    sharding_config: ShardingConfig,
) -> jax.Array:
    metadata = field_info.field.metadata
    tensor_sharding = metadata.get("tensor_sharding")
    min_size_to_shard = metadata.get("min_size_to_shard", 0)
    if tensor_sharding is None or array.size < min_size_to_shard:
        return array

    parts: list[str | None] = [None] * array.ndim
    sharding_order = getattr(field_info.owner, "sharding_order", None)

    tp_dim = next(
        (
            dim
            for dim in tensor_sharding.dims_to_try(sharding_order)
            if dim < array.ndim and array.shape[dim] % sharding_config.tensor_axis_size == 0
        ),
        None,
    )
    if tp_dim is not None:
        parts[tp_dim] = sharding_config.tensor_axis_name

    if sharding_config.fsdp:
        data_dim = next(
            (
                dim
                for dim in tensor_sharding.axes
                if dim != tp_dim and dim < array.ndim and array.shape[dim] % sharding_config.data_axis_size == 0
            ),
            None,
        )
        if data_dim is not None:
            parts[data_dim] = sharding_config.data_axis_name

    return jax.device_put(array, sharding_config.make_sharding(shd.PartitionSpec(*parts)))


def load_parameters[M: eqx.Module](
    selector: Callable[[M], SelectedTree],
    module: M,
    new_values: SelectedTree,
) -> M:
    old_values = selector(module)
    sharding_config = get_current_sharding_config()

    if _abstract_signature(old_values) != _abstract_signature(new_values):
        raise ValueError("load_parameters replacement must preserve the selected parameter structure")

    if sharding_config is None:
        return eqx.tree_at(selector, module, new_values, is_leaf=lambda leaf: leaf is None)

    def metadata_at(path: tuple[object, ...], leaf: object) -> FieldMetadataInfo | None:
        if not _is_leaf_array(leaf):
            return None
        return _field_info_at_path(module, path)

    metadata = selector(jtu.tree_map_with_path(metadata_at, module, is_leaf=lambda leaf: leaf is None))

    def shard_value(old_value: object, incoming_value: object, field_info: FieldMetadataInfo | None) -> object:
        if not _is_leaf_array(old_value):
            return incoming_value
        assert isinstance(field_info, FieldMetadataInfo)
        assert eqx.is_array(incoming_value)
        return _apply_parameter_sharding(incoming_value, field_info, sharding_config)

    loaded_values = jax.tree.map(
        shard_value,
        old_values,
        new_values,
        metadata,
        is_leaf=lambda leaf: leaf is None or _is_leaf_array(leaf) or isinstance(leaf, FieldMetadataInfo),
    )
    return eqx.tree_at(selector, module, loaded_values, is_leaf=lambda leaf: leaf is None)
