from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

import jax.numpy as jnp
from jax._src.api import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike

from lalamo.utils import MapDictValues, MapSequence

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
    Mapping[str, ArrayType | ParameterTree[ArrayType]] | Sequence[ArrayType | ParameterTree[ArrayType]]
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


type KeyTree = Mapping[str, str | KeyTree] | Sequence[str | KeyTree]


def _unflatten_keys(flat_keys: Mapping[str, str]) -> KeyTree:
    groups: dict[str, dict[str, str] | str] = defaultdict(dict)
    for subkey, full_key in flat_keys.items():
        match subkey.split(".", maxsplit=1):
            case [head]:
                groups[head] = full_key
            case [head, tail]:
                group = groups[head]
                assert isinstance(group, dict)
                group[tail] = full_key

    unflattened_groups: dict[str, KeyTree] = {}
    for subkey, group in groups.items():
        if isinstance(group, str):
            unflattened_groups[subkey] = group
        else:
            unflattened_groups[subkey] = _unflatten_keys(group)

    if any(key.isnumeric() for key in unflattened_groups):
        assert set(unflattened_groups.keys()) == set(map(str, range(len(unflattened_groups))))
        return [v for k, v in sorted(unflattened_groups.items(), key=lambda item: int(item[0]))]
    return unflattened_groups


def _recursive_map_dict[ArrayType: ArrayLike](
    key_tree: KeyTree | str,
    root_collection: Mapping[str, ArrayType],
) -> ParameterTree[ArrayType] | ArrayType:
    if isinstance(key_tree, str):
        return root_collection[key_tree]
    if isinstance(key_tree, Mapping):
        return MapDictValues(lambda subtree: _recursive_map_dict(subtree, root_collection), key_tree)
    if isinstance(key_tree, Sequence):
        return MapSequence(lambda subtree: _recursive_map_dict(subtree, root_collection), key_tree)


def unflatten_parameters[ArrayType: ArrayLike](flat_parameters: Mapping[str, ArrayType]) -> ParameterTree[ArrayType]:
    unflattened_keys = _unflatten_keys({k: k for k in flat_parameters})
    result = _recursive_map_dict(unflattened_keys, flat_parameters)
    assert not isinstance(result, (Array, ShapeDtypeStruct))
    return result


class ParameterPath(str):
    __slots__ = ()

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.split("."))

    def __truediv__(self, other: str | int) -> "ParameterPath":
        if not self:
            return ParameterPath(str(other))
        return ParameterPath(self + "." + str(other))
