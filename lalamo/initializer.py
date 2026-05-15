import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, Key

from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import (
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    ShapeDtypeMatrix,
    ShapeDtypeSpec,
    WeightMatrix,
)

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

    @abstractmethod
    def weight_matrix(
        self,
        output_dim: int,
        input_dim: int,
        mixture_size: int | None = None,
        *,
        is_sharded: bool = True,
    ) -> WeightMatrix: ...

    @abstractmethod
    def embedding_matrix(self, vocabulary_size: int, model_dim: int) -> EmbeddingMatrix: ...


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

    def weight_matrix(
        self,
        output_dim: int,
        input_dim: int,
        mixture_size: int | None = None,
        *,
        is_sharded: bool = True,
    ) -> ShapeDtypeMatrix:
        if mixture_size is None:
            mixture_dims = ()
        else:
            mixture_dims = (mixture_size,)
        return ShapeDtypeMatrix(
            spec=ShapeDtypeSpec(layout=Layout.OUTPUT_INPUT),
            is_sharded=is_sharded,
            mixture_dims=mixture_dims,
            input_dim=input_dim,
            output_dim=output_dim,
            dtype_=self.dtype,
        )

    def embedding_matrix(self, vocabulary_size: int, model_dim: int) -> ShapeDtypeMatrix:
        return ShapeDtypeMatrix(
            spec=ShapeDtypeSpec(layout=Layout.INPUT_OUTPUT),
            is_sharded=False,
            mixture_dims=(),
            input_dim=vocabulary_size,
            output_dim=model_dim,
            dtype_=self.dtype,
        )


@dataclass
class RandomInitializer(Initializer):
    key: Key[Array, ""] | None = field(default=None, kw_only=True)

    def normal(
        self,
        std: float,
        shape: tuple[int, ...],
        partition: tuple[ShardingAxis | None, ...] | None = None,
    ) -> Array:
        if self.key is None:
            raise ValueError("RandomInitializer requires a key.")
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

    def weight_matrix(
        self,
        output_dim: int,
        input_dim: int,
        mixture_size: int | None = None,
        *,
        is_sharded: bool = True,
    ) -> FullPrecisionMatrix:
        if mixture_size is None:
            mixture_dims = ()
        else:
            mixture_dims = (mixture_size,)
        std = 1.0 / math.sqrt(input_dim)
        weights = self.normal(std, (*mixture_dims, output_dim, input_dim))
        return FullPrecisionSpec(Layout.OUTPUT_INPUT).compress(weights, is_sharded=is_sharded)

    def embedding_matrix(self, vocabulary_size: int, model_dim: int) -> FullPrecisionMatrix:
        std = 1.0 / math.sqrt(model_dim)
        weights = self.normal(std, (model_dim, vocabulary_size), partition=(None, None))
        return FullPrecisionSpec(Layout.INPUT_OUTPUT).compress(weights, is_sharded=False)
