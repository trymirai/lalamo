from collections.abc import Callable, Iterable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.sharding as shd
from jax._src.api import ShapeDtypeStruct
from jax.tree import leaves_with_path
from jax.tree_util import keystr
from jaxtyping import Array, PyTree

from lalamo.modules.common import (
    FieldMetadataInfo,
    ShardingConfig,
    ShardingOrder,
    TensorSharding,
    find_field_metadata,
    get_current_sharding_config,
)

__all__ = [
    "load_parameters",
]


@dataclass(frozen=True)
class FieldShardingInfo:
    tensor_sharding: TensorSharding
    sharding_order: ShardingOrder | None
    min_size_to_shard: int


def find_field_sharding(module: eqx.Module, target: Array) -> FieldShardingInfo | None:
    field_info = find_field_metadata(module, target)
    if field_info is None:
        return None
    return _field_sharding_info(field_info)


def _field_sharding_info(field_info: FieldMetadataInfo) -> FieldShardingInfo | None:
    tensor_sharding = field_info.metadata.get("tensor_sharding")
    if tensor_sharding is None:
        return None
    return FieldShardingInfo(
        tensor_sharding,
        sharding_order=getattr(field_info.owner, "sharding_order", None),
        min_size_to_shard=field_info.metadata.get("min_size_to_shard", 0),
    )


def _apply_parameter_sharding(
    array: Array,
    info: FieldShardingInfo | None,
    sharding_config: ShardingConfig | None,
) -> Array:
    if sharding_config is None or info is None or array.size < info.min_size_to_shard:
        return array

    parts: list[str | None] = [None] * array.ndim

    tp_dim: int | None = None
    tp_axis_size = sharding_config.tensor_axis_size
    for dim in info.tensor_sharding.dims_to_try(info.sharding_order):
        if dim < array.ndim and array.shape[dim] % tp_axis_size == 0:
            parts[dim] = sharding_config.tensor_axis_name
            tp_dim = dim
            break

    if sharding_config.fsdp:
        dp_axis_size = sharding_config.data_axis_size
        for dim in info.tensor_sharding.axes:
            if dim != tp_dim and dim < array.ndim and array.shape[dim] % dp_axis_size == 0:
                parts[dim] = sharding_config.data_axis_name
                break

    pspec = shd.PartitionSpec(*parts)
    return jax.device_put(array, sharding_config.make_sharding(pspec))


def _get_name(leaf: PyTree, tree: PyTree) -> str:
    for path, value in leaves_with_path(tree):
        if value is leaf:
            return f"~{keystr(path)}"
    raise ValueError(f"Leaf {leaf} not found in tree {tree}")


def _check_compatible(old_value: PyTree, new_value: PyTree, module: eqx.Module) -> None:
    if isinstance(old_value, (Array, ShapeDtypeStruct)) and isinstance(new_value, Array):
        name = _get_name(old_value, module)
        if old_value.shape != new_value.shape:
            raise ValueError(
                f"Expected parameter {module}.{name} to have shape {old_value.shape}, got {new_value.shape}",
            )
        if old_value.dtype != new_value.dtype:
            raise ValueError(
                f"Expected parameter {module}.{name} to have dtype {old_value.dtype}, got {new_value.dtype}",
            )
    elif type(old_value) is not type(new_value):
        raise TypeError(f"Expected parameter of type {type(old_value)}, got {type(new_value)}")


def load_parameters[M: eqx.Module](
    selector: Callable[[M], Iterable[PyTree]],
    module: M,
    new_values: Iterable[PyTree],
) -> M:
    old_values = list(selector(module))
    new_values = list(new_values)
    sharding_config = get_current_sharding_config()

    casted_new_values = []

    for old_value, new_value in zip(old_values, new_values, strict=True):
        _check_compatible(old_value, new_value, module)
        if isinstance(old_value, (Array, ShapeDtypeStruct)) and isinstance(new_value, Array):
            new_value = new_value.astype(old_value.dtype)  # noqa: PLW2901
            new_value = _apply_parameter_sharding(  # noqa: PLW2901
                new_value,
                find_field_sharding(module, old_value),
                sharding_config,
            )
        casted_new_values.append(new_value)
    return eqx.tree_at(selector, module, casted_new_values, is_leaf=lambda x: x is None)
