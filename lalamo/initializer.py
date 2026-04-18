from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import jax
from jax import ShapeDtypeStruct
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, DTypeLike, Key

from .module import ShardingAxis

__all__ = [
    "EmptyInitializer",
    "Initializer",
    "RandomInitializer",
]


@dataclass
class Initializer(ABC):
    mesh: Mesh | None
    dtype: DTypeLike

    def partition_to_sharding(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> NamedSharding | None:
        if partition is None or self.mesh is None:
            return None
        if len(shape) != len(partition):
            raise ValueError(f"Shape {shape} and partition {partition} must have the same length")
        return NamedSharding(self.mesh, PartitionSpec(partition))

    @abstractmethod
    def normal(
        self,
        std: float,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array: ...

    @abstractmethod
    def ones(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array: ...

    @abstractmethod
    def zeros(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array: ...


class EmptyInitializer(Initializer):
    def _dummy_array(
        self,
        shape: int | tuple[int, ...],
        dtype: DTypeLike,
        partition: tuple[ShardingAxis | None, ...] | None,
    ) -> Array:
        if isinstance(shape, int):
            shape = (shape,)
        sharding = self.partition_to_sharding(shape, partition)
        return cast("Array", ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding))

    def normal(
        self,
        std: float,  # noqa: ARG002
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        return self._dummy_array(shape, self.dtype, partition)

    def ones(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        return self._dummy_array(shape, self.dtype, partition)

    def zeros(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        return self._dummy_array(shape, self.dtype, partition)


@dataclass
class RandomInitializer(Initializer):
    key: Key[Array, ""]

    def normal(
        self,
        std: float,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        sharding = self.partition_to_sharding(shape, partition)
        self.key, key = jax.random.split(self.key)
        return jax.random.normal(key, shape, self.dtype, out_sharding=sharding) * std

    def ones(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        sharding = self.partition_to_sharding(shape, partition)
        return jnp.ones(shape, self.dtype, out_sharding=sharding)

    def zeros(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        sharding = self.partition_to_sharding(shape, partition)
        return jnp.zeros(shape, self.dtype, out_sharding=sharding)
