import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, Key

from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import LogicalAxis, ShardingConfig
from lalamo.weight_matrix import (
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    ShapeDtypeMatrix,
    ShapeDtypeSpec,
    WeightMatrix,
)

__all__ = [
    "EmptyInitializer",
    "Initializer",
    "RandomInitializer",
]


@dataclass
class Initializer(ABC):
    dtype: DTypeLike
    sharding_config: ShardingConfig

    def _partition_to_sharding(
        self,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
    ) -> NamedSharding:
        if partition is None:
            return self.sharding_config.resolve_sharding((None,) * len(shape))
        if len(shape) != len(partition):
            raise ValueError(f"Shape {shape} and partition {partition} must have the same length")
        return self.sharding_config.resolve_sharding(partition)

    @abstractmethod
    def normal(
        self,
        std: float,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
    ) -> Array: ...

    @abstractmethod
    def ones(
        self,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
    ) -> Array: ...

    @abstractmethod
    def zeros(
        self,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
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
        partition: tuple[LogicalAxis | None, ...] | None,
    ) -> Array:
        sharding = self._partition_to_sharding(shape, partition)
        return dummy_array(shape, dtype, sharding)

    def normal(
        self,
        std: float,  # noqa: ARG002
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
    ) -> Array:
        return self._dummy_array(shape, self.dtype, partition)

    def ones(
        self,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
    ) -> Array:
        return self._dummy_array(shape, self.dtype, partition)

    def zeros(
        self,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
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
        return ShapeDtypeSpec(layout=Layout.OUTPUT_INPUT).compress(
            self._dummy_array(
                (*mixture_dims, output_dim, input_dim),
                self.dtype,
                None,
            ),
            sharding_config=self.sharding_config,
            is_sharded=is_sharded,
        )

    def embedding_matrix(self, vocabulary_size: int, model_dim: int) -> ShapeDtypeMatrix:
        return ShapeDtypeSpec(layout=Layout.INPUT_OUTPUT).compress(
            self._dummy_array(
                (model_dim, vocabulary_size),
                self.dtype,
                (None, None),
            ),
            sharding_config=self.sharding_config,
            is_sharded=False,
        )


@dataclass
class RandomInitializer(Initializer):
    key: Key[Array, ""] | None = field(default=None, kw_only=True)

    def normal(
        self,
        std: float,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
    ) -> Array:
        if self.key is None:
            raise ValueError("RandomInitializer requires a key.")
        sharding = self._partition_to_sharding(shape, partition)
        self.key, key = jax.random.split(self.key)
        values = jax.random.normal(key, shape, self.dtype) * std
        return jax.device_put(values, sharding)

    def ones(
        self,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
    ) -> Array:
        sharding = self._partition_to_sharding(shape, partition)
        values = jnp.ones(shape, self.dtype)
        return jax.device_put(values, sharding)

    def zeros(
        self,
        shape: tuple[int, ...],
        partition: tuple[LogicalAxis | None, ...] | None = None,
    ) -> Array:
        sharding = self._partition_to_sharding(shape, partition)
        values = jnp.zeros(shape, self.dtype)
        return jax.device_put(values, sharding)

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
        weights = self.normal(
            std,
            (*mixture_dims, output_dim, input_dim),
        )
        return FullPrecisionSpec(Layout.OUTPUT_INPUT).compress(
            weights,
            sharding_config=self.sharding_config,
            is_sharded=is_sharded,
        )

    def embedding_matrix(self, vocabulary_size: int, model_dim: int) -> FullPrecisionMatrix:
        std = 1.0 / math.sqrt(model_dim)
        weights = self.normal(std, (model_dim, vocabulary_size), partition=(None, None))
        return FullPrecisionSpec(Layout.INPUT_OUTPUT).compress(
            weights,
            sharding_config=self.sharding_config,
            is_sharded=False,
        )
