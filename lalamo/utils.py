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

import einops
import jax.numpy as jnp
import torch
import torch.utils.dlpack
from jaxtyping import Array

__all__ = [
    "MapDict",
    "MapSequence",
    "jax_to_torch",
    "jax_uint4_to_packed_uint8",
    "torch_to_jax",
]


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
class MapDict[K, OldV, NewV](Mapping[K, NewV]):
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


@torch.no_grad()
def _torch_to_jax_bfloat16(tensor: torch.Tensor) -> Array:
    # Credit: https://github.com/jax-ml/ml_dtypes/issues/81#issuecomment-2399636232
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Trying to convert non-bfloat16 tensor to bfloat16")
    intermediate_tensor = tensor.view(torch.uint16)
    return jnp.array(intermediate_tensor).view("bfloat16")


def torch_to_jax(array: torch.Tensor) -> Array:
    array = array.detach().cpu()
    if array.dtype == torch.bfloat16:
        return _torch_to_jax_bfloat16(array)
    return jnp.array(array.numpy())


def jax_to_torch(array: Array) -> torch.Tensor:
    if array.dtype == jnp.bfloat16:
        intermediate_array = array.view(jnp.uint16)
        return torch.utils.dlpack.from_dlpack(intermediate_array).view(torch.bfloat16)
    return torch.utils.dlpack.from_dlpack(array)


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
