from collections.abc import Callable, Iterable

import equinox as eqx
import jax
import jax.sharding as shd
from jax.tree import leaves_with_path
from jax.tree_util import keystr
from jaxtyping import PyTree

from lalamo.modules.common import (
    FieldMetadataInfo,
    ShardingConfig,
    _is_leaf_array,
    field_metadata_by_leaf_id,
    get_current_sharding_config,
)

__all__ = [
    "load_parameters",
]


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
    selector: Callable[[M], Iterable[PyTree]],
    module: M,
    new_values: Iterable[PyTree],
) -> M:
    old_values = list(selector(module))
    new_values = list(new_values)
    sharding_config = get_current_sharding_config()
    metadata_by_leaf_id = field_metadata_by_leaf_id(module) if sharding_config is not None else None
    leaf_names_by_id = {id(value): f"~{keystr(path)}" for path, value in leaves_with_path(module)}
    casted_new_values: list[PyTree] = []

    for old_value, incoming_value in zip(old_values, new_values, strict=True):
        leaf_name = leaf_names_by_id.get(id(old_value), "<selected leaf>")
        if _is_leaf_array(old_value) and eqx.is_array(incoming_value):
            if old_value.shape != incoming_value.shape:
                raise ValueError(
                    f"Expected parameter {module}.{leaf_name} to have shape {old_value.shape},"
                    f" got {incoming_value.shape}",
                )
            if old_value.dtype != incoming_value.dtype:
                raise ValueError(
                    f"Expected parameter {module}.{leaf_name} to have dtype {old_value.dtype},"
                    f" got {incoming_value.dtype}",
                )
            loaded_value = incoming_value
            if sharding_config is not None:
                assert metadata_by_leaf_id is not None
                field_info = metadata_by_leaf_id.get(id(old_value))
                assert field_info is not None, f"Missing field metadata for sharded parameter {leaf_name}"
                loaded_value = _apply_parameter_sharding(
                    loaded_value,
                    field_info,
                    sharding_config,
                )
            casted_new_values.append(loaded_value)
            continue
        if type(old_value) is not type(incoming_value):
            raise TypeError(f"Expected parameter of type {type(old_value)}, got {type(incoming_value)}")
        casted_new_values.append(incoming_value)
    return eqx.tree_at(selector, module, casted_new_values, is_leaf=lambda x: x is None)
