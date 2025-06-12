from collections.abc import Callable, Iterable

import equinox as eqx
from jax.tree import leaves_with_path
from jax.tree_util import keystr
from jaxtyping import Array, PyTree

__all__ = [
    "load_parameters",
]


def _get_name(leaf: PyTree, tree: PyTree) -> str:
    for path, value in leaves_with_path(tree):
        if value is leaf:
            return f"~{keystr(path)}"
    raise ValueError(f"Leaf {leaf} not found in tree {tree}")


def _check_compatible(old_value: PyTree, new_value: PyTree, module: eqx.Module) -> None:
    if isinstance(old_value, Array) and isinstance(new_value, Array):
        name = _get_name(old_value, module)
        if old_value.shape != new_value.shape:
            raise ValueError(f"Expected parameter {name} to have shape {old_value.shape}, got {new_value.shape}")
        if old_value.dtype != new_value.dtype:
            raise ValueError(f"Expected parameter {name} to have dtype {old_value.dtype}, got {new_value.dtype}")
    elif type(old_value) is not type(new_value):
        raise TypeError(f"Expected parameter of type {type(old_value)}, got {type(new_value)}")


def load_parameters[M: eqx.Module](
    selector: Callable[[M], Iterable[PyTree]],
    module: M,
    new_values: Iterable[PyTree],
) -> M:
    old_values = list(selector(module))
    new_values = list(new_values)
    casted_new_values = []
    for old_value, new_value in zip(old_values, new_values, strict=True):
        _check_compatible(old_value, new_value, module)
        if isinstance(old_value, Array) and isinstance(new_value, Array):
            casted_new_values.append(new_value.astype(old_value.dtype))
        else:
            casted_new_values.append(new_value)
    return eqx.tree_at(selector, module, casted_new_values, is_leaf=lambda x: x is None)
