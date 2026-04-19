from collections.abc import Callable, Iterable

import equinox as eqx
from jax import Array, ShapeDtypeStruct
from jaxtyping import PyTree

__all__ = [
    "load_parameters",
]


def _check_compatible(old_value: PyTree, new_value: PyTree, module: eqx.Module) -> None:
    if isinstance(old_value, (Array, ShapeDtypeStruct)) and isinstance(new_value, Array):
        if old_value.shape != new_value.shape:
            raise ValueError(
                f"Expected parameter in {type(module).__name__} "
                f"to have shape {old_value.shape}, got {new_value.shape}",
            )
        return
    if old_value is None or new_value is None:
        if old_value is not None or new_value is not None:
            raise TypeError(f"Expected parameter of type {type(old_value)}, got {type(new_value)}")
        return
    if type(old_value) is not type(new_value):
        raise TypeError(f"Expected parameter of type {type(old_value)}, got {type(new_value)}")


def load_parameters[M: eqx.Module](
    selector: Callable[[M], Iterable[PyTree]],
    module: M,
    new_values: Iterable[PyTree],
) -> M:
    old_values = list(selector(module))
    casted_new_values = []
    for old_value, new_value in zip(old_values, new_values, strict=True):
        _check_compatible(old_value, new_value, module)
        if isinstance(old_value, (Array, ShapeDtypeStruct)) and isinstance(new_value, Array):
            casted_value: PyTree = new_value.astype(old_value.dtype)
        else:
            casted_value = new_value
        casted_new_values.append(casted_value)
    return eqx.tree_at(selector, module, casted_new_values, is_leaf=lambda x: x is None)
