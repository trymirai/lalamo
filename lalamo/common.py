from collections import defaultdict
from collections.abc import Mapping
from typing import cast

import jax.numpy as jnp
from jax._src.api import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike

__all__ = [
    "DEFAULT_PRECISION",
    "ArrayLike",
    "ParameterPath",
    "ParameterTree",
    "dummy_array",
    "flatten_parameters",
    "unflatten_parameters",
]

DEFAULT_PRECISION: DTypeLike = jnp.bfloat16


type ArrayLike = Array | ShapeDtypeStruct


type ParameterTree[ArrayType: ArrayLike] = (
    Mapping[str, ArrayType | ParameterTree[ArrayType]] | list[ArrayType | ParameterTree[ArrayType]]
)


def dummy_array(shape: int | tuple[int, ...], dtype: DTypeLike) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    return cast("Array", ShapeDtypeStruct(shape=shape, dtype=dtype))


def flatten_parameters[ArrayType: ArrayLike](nested_parameters: ParameterTree[ArrayType]) -> dict[str, ArrayType]:
    result: dict[str, ArrayType] = {}
    if not isinstance(nested_parameters, Mapping):
        nested_parameters = {str(i): value for i, value in enumerate(nested_parameters)}
    for key, value in nested_parameters.items():
        key_path = ParameterPath(key)
        if isinstance(value, (Array, ShapeDtypeStruct)):
            result[key_path] = value
        else:
            update: dict[str, ArrayType] = {
                str(key_path / subkey): subvalue for subkey, subvalue in flatten_parameters(value).items()
            }
            result.update(update)
    return result


def unflatten_parameters[ArrayType: ArrayLike](flat_parameters: dict[str, ArrayType]) -> ParameterTree[ArrayType]:
    groups: dict[str, dict[str, ArrayType] | ArrayType] = defaultdict(dict)
    for key, value in flat_parameters.items():
        match key.split(".", maxsplit=1):
            case [head]:
                groups[head] = value
            case [head, tail]:
                group = groups[head]
                assert isinstance(group, dict)
                group[tail] = value

    unflattened_groups: dict[str, ParameterTree[ArrayType] | ArrayType] = {}
    for key, value in groups.items():
        if isinstance(value, (Array, ShapeDtypeStruct)):
            unflattened_groups[key] = value
        else:
            unflattened_groups[key] = unflatten_parameters(value)

    if any(key.isnumeric() for key in unflattened_groups):
        assert set(unflattened_groups.keys()) == set(map(str, range(len(unflattened_groups))))
        return list(unflattened_groups.values())
    return unflattened_groups


class ParameterPath(str):
    __slots__ = ()

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.split("."))

    def __truediv__(self, other: str | int) -> "ParameterPath":
        if not self:
            return ParameterPath(str(other))
        return ParameterPath(self + "." + str(other))
