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
from dataclasses import dataclass
from typing import overload

__all__ = [
    "LazyDict",
    "MapCollection",
    "MapDictValues",
    "MapIterable",
    "MapSequence",
    "MapValuesView",
    "MapView",
    "MappedValues",
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


@dataclass(frozen=True)
class MappedValues[K, OldV, NewV](Mapping[K, NewV]):
    collection: Mapping[K, OldV]
    value_map: Callable[[K, OldV], NewV]

    def __getitem__(self, key: K) -> NewV:
        return self.value_map(key, self.collection[key])

    def __iter__(self) -> Iterator[K]:
        return iter(self.collection)

    def __len__(self) -> int:
        return len(self.collection)
