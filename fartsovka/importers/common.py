from collections.abc import Callable, Iterable

import equinox as eqx
from jax.tree import leaves_with_path
from jax.tree_util import keystr
from jaxtyping import Array, PyTree


def get_name(leaf: PyTree, tree: PyTree) -> str:
    for path, value in leaves_with_path(tree):
        if value is leaf:
            return f"~{keystr(path)}"
    raise ValueError(f"Leaf {leaf} not found in tree {tree}")


def load_parameters[M: eqx.Module](
    selector: Callable[[M], Iterable[PyTree]],
    module: M,
    new_values: Iterable[PyTree],
) -> M:
    old_values = list(selector(module))
    new_values = list(new_values)
    casted_new_values = []
    for old_value, new_value in zip(old_values, new_values, strict=True):
        if isinstance(old_value, Array) and isinstance(new_value, Array):
            if old_value.shape != new_value.shape:
                name = get_name(old_value, module)
                raise ValueError(f"Expected parameter {name} to have shape {old_value.shape}, got {new_value.shape}")
            casted_new_values.append(new_value.astype(old_value.dtype))
        elif type(old_value) is type(new_value):
            casted_new_values.append(new_value)
        else:
            name = get_name(old_value, module)
            raise TypeError(f"Expected parameter {name} to have type {type(old_value)}, got {type(new_value)}")
    return eqx.tree_at(selector, module, casted_new_values)


class WeightsPath(str):
    __slots__ = ()

    def __truediv__(self, other: str | int) -> "WeightsPath":
        if not self:
            return WeightsPath(str(other))
        return WeightsPath(self + "." + str(other))
