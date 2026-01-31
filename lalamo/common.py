import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

import jax
import jax.numpy as jnp
from jax._src.api import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike

from lalamo.utils import MapDictValues, MapSequence

__all__ = [
    "DEFAULT_PRECISION",
    "ArrayLike",
    "LalamoWarning",
    "ParameterPath",
    "ParameterTree",
    "dummy_array",
    "flatten_parameters",
    "require_array",
    "require_tree",
    "unflatten_parameters",
]

DEFAULT_PRECISION: DTypeLike = jnp.bfloat16


class LalamoWarning(UserWarning):
    """Custom warning class for Lalamo-specific warnings."""


type ArrayLike = Array | ShapeDtypeStruct


type ParameterTree[ArrayType: ArrayLike] = (
    Mapping[str, ArrayType | ParameterTree[ArrayType]] | Sequence[ArrayType | ParameterTree[ArrayType]]
)


def require_array[ArrayType: ArrayLike](value: ArrayType | ParameterTree[ArrayType]) -> ArrayType:
    assert not isinstance(value, (Mapping, Sequence))
    return value


def require_tree[ArrayType: ArrayLike](value: ArrayType | ParameterTree[ArrayType]) -> ParameterTree[ArrayType]:
    assert not isinstance(value, (Array, ShapeDtypeStruct))
    return value


def dummy_array(shape: int | tuple[int, ...], dtype: DTypeLike) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    return cast("Array", ShapeDtypeStruct(shape=shape, dtype=dtype))


def flatten_parameters[ArrayType: ArrayLike](nested_parameters: ParameterTree[ArrayType]) -> dict[str, ArrayType]:
    result: dict[str, ArrayType] = {}
    if not isinstance(nested_parameters, Mapping):
        nested_parameters = {str(i): value for i, value in enumerate(nested_parameters)}
    for key, value in nested_parameters.items():
        value = cast("ArrayType | ParameterTree[ArrayType]", value)
        key_path = ParameterPath(key)
        if isinstance(value, (Array, ShapeDtypeStruct)):
            result[key_path] = cast("ArrayType", value)
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


def get_default_device_bytes() -> int | None:
    dynamic_allocate = False

    preallocate = os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE", "")
    dynamic_allocate |= preallocate.strip().lower() in {"0", "false", "no", "off"}

    allocator = os.getenv("XLA_PYTHON_CLIENT_ALLOCATOR", "")
    dynamic_allocate |= allocator.strip().lower() in {"platform", "cuda_malloc_async"}

    if dynamic_allocate:
        return None

    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None or "bytes_limit" not in memory_stats:
        return None

    mem_fraction_raw = os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "")
    try:
        mem_fraction = float(mem_fraction_raw)
    except ValueError:
        mem_fraction = 0.75  # jax default https://docs.jax.dev/en/latest/gpu_memory_allocation.html

    # 500mb is seemingly the usually observed overhead; this tries to match the actual capacity of the gpu
    # so it should correspond to something you'd see in nvidia-smi
    memory_limit = memory_stats["bytes_limit"] / min(mem_fraction, 1.0) + (500 * 1000 * 1000)

    return memory_limit


def get_usable_memory_from_bytes(limit_bytes: int) -> int:
    return int(limit_bytes * 0.93)
