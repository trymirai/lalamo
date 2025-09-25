from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MappingView,
    Sequence,
    ValuesView,
)
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import einops
import jax.numpy as jnp
from jaxtyping import Array
from safetensors import safe_open

__all__ = [
    "MapDictValues",
    "MapSequence",
    "jax_uint4_to_packed_uint8",
    "open_safetensors",
]


@dataclass(frozen=True)
class LazyDict[K, V](Mapping[K, V]):
    stored_keys: set[K]
    getter: Callable[[K], V]

    def __getitem__(self, key: K) -> V:
        if key not in self.stored_keys:
            raise KeyError(key)
        return self.getter(key)

    def __iter__(self) -> Iterator[K]:
        return iter(self.stored_keys)

    def __len__(self) -> int:
        return len(self.stored_keys)


@contextmanager
def open_safetensors(filename: Path | str) -> Iterator[Mapping[str, Array]]:
    with safe_open(filename, framework="flax") as safetensors_nonsense:
        yield LazyDict(set(safetensors_nonsense.keys()), safetensors_nonsense.get_tensor)


@dataclass(frozen=True)
class MapIterable[OldT, NewT](Iterable[NewT]):
    map_func: Callable[[OldT], NewT]
    collection: Iterable[OldT]

    def __iter__(self) -> Iterator[NewT]:
        return map(self.map_func, self.collection)


@dataclass(frozen=True)
class MapCollection[OldT, NewT](MapIterable[OldT, NewT], Collection[NewT]):
    collection: Collection[OldT]

    def __contains__(self, item: object) -> bool:
        return any(self.map_func(elem) == item for elem in self.collection)

    def __len__(self) -> int:
        return len(self.collection)


@dataclass(frozen=True)
class MapSequence[OldT, NewT](MapCollection[OldT, NewT], Sequence[NewT]):
    collection: Sequence[OldT]

    @overload
    def __getitem__(self, index: int) -> NewT: ...

    @overload
    def __getitem__(self, index: slice) -> list[NewT]: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.map_func(item) for item in self.collection[index]]
        return self.map_func(self.collection[index])


@dataclass(frozen=True)
class MapView[OldT, NewT](MapCollection[OldT, NewT], MappingView):
    pass


@dataclass(frozen=True)
class MapValuesView[OldT, NewT](MapCollection[OldT, NewT], ValuesView):
    pass


@dataclass(frozen=True)
class MapDictValues[K, OldV, NewV](Mapping[K, NewV]):
    value_map: Callable[[OldV], NewV]
    collection: Mapping[K, OldV]

    def __getitem__(self, key: K) -> NewV:
        return self.value_map(self.collection[key])

    def __iter__(self) -> Iterator[K]:
        return iter(self.collection)

    def keys(self) -> KeysView[K]:
        return self.collection.keys()

    def values(self) -> MapValuesView[OldV, NewV]:
        return MapValuesView(self.value_map, self.collection.values())

    def __len__(self) -> int:
        return len(self.collection)


def jax_uint4_to_packed_uint8(array: Array) -> Array:
    if array.dtype != jnp.uint4:
        raise ValueError(f"Input array must have dtype jnp.uint4, but got {array.dtype}")

    if not array.shape:
        raise ValueError("Input array cannot be a scalar and must have at least one dimension.")

    *_, last_dim = array.shape
    if last_dim % 2 != 0:
        raise ValueError(f"The last dimension of the input array must be even, but got shape {array.shape}")

    low_nibbles, high_nibbles = einops.rearrange(
        array.astype(jnp.uint8),
        "... (dim_half two) -> two ... dim_half",
        two=2,
    )

    packed = (high_nibbles << 4) | low_nibbles

    return packed.astype(jnp.uint8)
