from collections.abc import Callable, Iterable

import equinox as eqx
import jax
import jax.sharding as shd
from jax._src.api import ShapeDtypeStruct
from jax.tree import leaves_with_path
from jax.tree_util import keystr
from jaxtyping import Array, PyTree

from lalamo.modules.common import find_field_metadata, get_current_sharding_config

__all__ = [
    "load_parameters",
]


def load_parameters[M: eqx.Module](
    selector: Callable[[M], Iterable[PyTree]],
    module: M,
    new_values: Iterable[PyTree],
) -> M:
    old_values = list(selector(module))
    new_values = list(new_values)
    sharding_config = get_current_sharding_config()
    module_leaves = tuple(leaves_with_path(module))

    def leaf_name(target: object) -> str:
        for path, value in module_leaves:
            if value is target:
                return f"~{keystr(path)}"
        raise ValueError(f"Leaf {target} not found in tree {module}")

    casted_new_values = []

    for old_value, incoming_value in zip(old_values, new_values, strict=True):
        if isinstance(old_value, (Array, ShapeDtypeStruct)) and isinstance(incoming_value, Array):
            if old_value.shape != incoming_value.shape:
                raise ValueError(
                    f"Expected parameter {module}.{leaf_name(old_value)} to have shape {old_value.shape},"
                    f" got {incoming_value.shape}",
                )
            if old_value.dtype != incoming_value.dtype:
                raise ValueError(
                    f"Expected parameter {module}.{leaf_name(old_value)} to have dtype {old_value.dtype},"
                    f" got {incoming_value.dtype}",
                )
            loaded_value = incoming_value
            if sharding_config is not None:
                field_info = find_field_metadata(module, old_value)
                if field_info is not None:
                    metadata = field_info.field.metadata
                    tensor_sharding = metadata.get("tensor_sharding")
                    min_size_to_shard = metadata.get("min_size_to_shard", 0)
                    if tensor_sharding is not None and loaded_value.size >= min_size_to_shard:
                        parts: list[str | None] = [None] * loaded_value.ndim
                        sharding_order = getattr(field_info.owner, "sharding_order", None)

                        tp_dim: int | None = None
                        for dim in tensor_sharding.dims_to_try(sharding_order):
                            if (
                                dim < loaded_value.ndim
                                and loaded_value.shape[dim] % sharding_config.tensor_axis_size == 0
                            ):
                                parts[dim] = sharding_config.tensor_axis_name
                                tp_dim = dim
                                break

                        if sharding_config.fsdp:
                            for dim in tensor_sharding.axes:
                                if (
                                    dim != tp_dim
                                    and dim < loaded_value.ndim
                                    and loaded_value.shape[dim] % sharding_config.data_axis_size == 0
                                ):
                                    parts[dim] = sharding_config.data_axis_name
                                    break

                        loaded_value = jax.device_put(
                            loaded_value,
                            sharding_config.make_sharding(shd.PartitionSpec(*parts)),
                        )
            casted_new_values.append(loaded_value)
            continue
        if type(old_value) is not type(incoming_value):
            raise TypeError(f"Expected parameter of type {type(old_value)}, got {type(incoming_value)}")
        casted_new_values.append(incoming_value)
    return eqx.tree_at(selector, module, casted_new_values, is_leaf=lambda x: x is None)
