from collections.abc import Callable, Iterable

import equinox as eqx
from jaxtyping import Array, PyTree


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
                raise ValueError(f"Expected parameter of shape {old_value.shape}, got {new_value.shape}")
            casted_new_values.append(new_value.astype(old_value.dtype))
        elif type(old_value) is type(new_value):
            casted_new_values.append(new_value)
        else:
            raise TypeError(f"Expected parameter of type {type(old_value)}, got {type(new_value)}")
    return eqx.tree_at(selector, module, casted_new_values)
