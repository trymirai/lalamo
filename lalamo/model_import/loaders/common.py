import dataclasses
from collections.abc import Callable, Iterable

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax._src.api import ShapeDtypeStruct
from jax.tree import leaves_with_path
from jax.tree_util import keystr
from jaxtyping import Array, PyTree

from lalamo.modules.common import (
    FieldShardingInfo,
    apply_parameter_sharding,
    get_current_sharding_config,
)

__all__ = [
    "load_parameters",
]


def find_field_sharding(module: eqx.Module, target: object) -> FieldShardingInfo | None:
    flat_with_path, _ = jtu.tree_flatten_with_path(module, is_leaf=lambda x: x is target)

    for path, leaf in flat_with_path:
        if leaf is not target:
            continue

        current_node: eqx.Module | object = module
        owner: eqx.Module = module
        owner_field: dataclasses.Field | None = None

        for key in path:
            if isinstance(key, jtu.GetAttrKey):
                assert isinstance(current_node, eqx.Module)
                owner = current_node
                owner_field = next((f for f in dataclasses.fields(current_node) if f.name == key.name), None)
                current_node = getattr(current_node, key.name)
            elif isinstance(key, jtu.SequenceKey):
                current_node = current_node[key.idx]  # type: ignore[index]
            elif isinstance(key, jtu.DictKey):
                current_node = current_node[key.key]  # type: ignore[index]
            else:
                raise TypeError(f"Unexpected key type: at {path}, key {key}")

        if owner_field is None:
            raise ValueError(f"Attempting to shard root module ({module}), or field on {path} not found")

        tensor_sharding = owner_field.metadata.get("tensor_sharding")
        if tensor_sharding is None:
            return None

        return FieldShardingInfo(
            tensor_sharding,
            sharding_order=getattr(owner, "sharding_order", None),
            min_size_to_shard=owner_field.metadata.get("min_size_to_shard", 0),
        )

    return None


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

        if sharding_config is not None:
            info = find_field_sharding(module, old_value)
            new_value = jax.tree.map(  # noqa: PLW2901
                lambda x, info=info: apply_parameter_sharding(x, info, sharding_config) if eqx.is_array(x) else x,
                new_value,
            )

        casted_new_values.append(new_value)
    return eqx.tree_at(selector, module, casted_new_values, is_leaf=lambda x: x is None)
