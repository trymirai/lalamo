from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, Key

from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding

from .module import ShardingAxis

__all__ = [
    "EmptyInitializer",
    "Initializer",
    "RandomInitializer",
]


@dataclass
class Initializer(ABC):
    dtype: DTypeLike

    def _partition_to_sharding(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> NamedSharding | None:
        if partition is not None and len(shape) != len(partition):
            raise ValueError(f"Shape {shape} and partition {partition} must have the same length")
        return make_sharding(partition)

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


@dataclass
class EmptyInitializer(Initializer):
    def _dummy_array(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        partition: tuple[ShardingAxis | None, ...] | None,
    ) -> Array:
        sharding = self._partition_to_sharding(shape, partition)
        return dummy_array(shape, dtype, sharding)

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
        sharding = self._partition_to_sharding(shape, partition)
        self.key, key = jax.random.split(self.key)
        return jax.random.normal(key, shape, self.dtype, out_sharding=sharding) * std

    def ones(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        sharding = self._partition_to_sharding(shape, partition)
        return jnp.ones(shape, self.dtype, out_sharding=sharding)

    def zeros(
        self,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        sharding = self._partition_to_sharding(shape, partition)
        return jnp.zeros(shape, self.dtype, out_sharding=sharding)
