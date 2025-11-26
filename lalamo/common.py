from abc import ABC, ABCMeta
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast
from weakref import WeakSet

import jax.numpy as jnp
from jax._src.api import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike

from lalamo.utils import MapDictValues, MapSequence

__all__ = [
    "DEFAULT_PRECISION",
    "JSON",
    "ArrayLike",
    "ParameterPath",
    "ParameterTree",
    "RegistryABC",
    "RegistryMeta",
    "dummy_array",
    "flatten_parameters",
    "unflatten_parameters",
]

DEFAULT_PRECISION: DTypeLike = jnp.bfloat16


type ArrayLike = Array | ShapeDtypeStruct


type ParameterTree[ArrayType: ArrayLike] = (
    Mapping[str, ArrayType | ParameterTree[ArrayType]] | Sequence[ArrayType | ParameterTree[ArrayType]]
)

type JSON = str | int | float | bool | None | dict[str, JSON] | list[JSON]


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


class RegistryMeta(ABCMeta):
    """
    Metaclass that tracks, for each subclass of RegistryABC, a per-class WeakSet
    of descendants (classes that have it in their MRO) while excluding any class
    that directly lists `RegistryABC` among its bases.
    """

    _REG_ATTR: str = "__registry_descendants__"
    _ROOT: type["RegistryABC"] | None = None

    def __init__(
        cls: type,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        # Give *every* class its own WeakSet (shadow any inherited attribute)
        setattr(cls, RegistryMeta._REG_ATTR, WeakSet())

        # Detect and remember the root exactly once
        if RegistryMeta._ROOT is None and name == "RegistryABC":
            RegistryMeta._ROOT = cls  # type: ignore[assignment]
            return

        root = RegistryMeta._ROOT
        if root is None:
            # Extremely early import edge-case; nothing to register yet
            return

        # Exclude classes that directly list the root among bases
        if any(b is root for b in cls.__bases__):
            return

        # Register this class on all qualifying ancestors below the root
        for ancestor in cls.__mro__[1:]:
            if isinstance(ancestor, RegistryMeta) and issubclass(ancestor, root):
                getattr(ancestor, RegistryMeta._REG_ATTR).add(cls)


class RegistryABC(ABC, metaclass=RegistryMeta):
    """
    Abstract base tracked by RegistryMeta.

    Any class defined as `class AbstractFoo(RegistryABC)` will expose a
    class method `AbstractFoo.__get_descendants__()` that returns a list of
    all classes having AbstractFoo in their MRO *except* those that directly
    include `RegistryABC` among their bases.
    """

    @classmethod
    def __descendants__(cls) -> tuple[type, ...]:
        reg: WeakSet[type] = getattr(cls, RegistryMeta._REG_ATTR)  # noqa: SLF001
        return tuple(reg)
