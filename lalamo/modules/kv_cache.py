from abc import abstractmethod
from typing import Protocol, Self

import equinox as eqx
import jax.numpy as jnp
from jax.lax import dynamic_update_slice_in_dim
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.common import ParameterDict

__all__ = ["KVCacheLayer"]


class KVCacheLayer(Protocol):
    @property
    @abstractmethod
    def keys(self) -> Float[Array, "tokens groups head_channels"]: ...

    @property
    @abstractmethod
    def values(self) -> Float[Array, "tokens groups head_channels"]: ...

    @abstractmethod
    def extend(
        self,
        new_keys: Float[Array, "new_tokens groups head_channels"],
        new_values: Float[Array, "new_tokens groups head_channels"],
    ) -> "KVCacheLayer": ...

    @classmethod
    @abstractmethod
    def empty(cls, num_groups: int, head_dim: int, dtype: DTypeLike) -> Self: ...

    def export(self) -> ParameterDict:
        return ParameterDict(
            keys=self.keys,
            values=self.values,
        )


class DynamicKVCacheLayer(eqx.Module, KVCacheLayer):
    keys: Float[Array, "tokens groups head_channels"]
    values: Float[Array, "tokens groups head_channels"]

    def extend(
        self,
        new_keys: Float[Array, "new_tokens groups head_channels"],
        new_values: Float[Array, "new_tokens groups head_channels"],
    ) -> "KVCacheLayer":
        raise NotImplementedError

    @classmethod
    def empty(cls, num_groups: int, head_dim: int, dtype: DTypeLike) -> Self:
        raise NotImplementedError

    def export(self) -> ParameterDict:
        return ParameterDict(
            keys=self.keys,
            values=self.values,
        )


class StaticKVCacheLayer(eqx.Module, KVCacheLayer):
    sequence_length: Int[Array, ""]
    keys_buffer: Float[Array, "capacity groups head_channels"]
    values_buffer: Float[Array, "capacity groups head_channels"]

    def __post_init__(self) -> None:
        if self.keys_buffer.ndim != 3:
            raise ValueError("Key and value buffers must have 3 dimensions: capacity, groups, head_channels")
        if self.keys_buffer.shape != self.values_buffer.shape:
            raise ValueError("Keys and values buffers must have the same shape")
        if self.keys_buffer.dtype != self.values_buffer.dtype:
            raise ValueError("Keys and values buffers must have the same dtype")

    @property
    def keys(self) -> Float[Array, "tokens groups head_channels"]:
        return self.keys_buffer[: self.sequence_length]

    @property
    def values(self) -> Float[Array, "tokens groups head_channels"]:
        return self.values_buffer[: self.sequence_length]

    @property
    def capacity(self) -> int:
        result, _, _ = self.keys_buffer.shape
        return result

    def extend(
        self,
        new_keys: Float[Array, "tokens groups head_channels"],
        new_values: Float[Array, "tokens groups head_channels"],
    ) -> "KVCacheLayer":
        if new_keys.shape != new_values.shape:
            raise ValueError("Keys and values must have the same shape")
        num_new_tokens, new_num_groups, new_head_dim = new_keys.shape
        old_capacity, old_num_groups, old_head_dim = self.keys_buffer.shape
        if new_num_groups != old_num_groups or new_head_dim != old_head_dim:
            raise ValueError("New keys and values must have the same number of groups and head dimensions")

        if old_capacity == 0:
            return KVCacheLayer(
                sequence_length=jnp.array(num_new_tokens),
                keys_buffer=new_keys,
                values_buffer=new_values,
            )

        old_sequence_length = self.sequence_length
        new_sequence_length = old_sequence_length + num_new_tokens

        if new_sequence_length > old_capacity:
            new_capacity = max(old_capacity * 2, new_sequence_length)
            _, num_groups, head_dim = self.keys_buffer.shape
            old_keys_buffer = jnp.empty((new_capacity, num_groups, head_dim), dtype=self.keys_buffer.dtype)
            old_keys_buffer = old_keys_buffer.at[:old_capacity].set(self.keys_buffer)
            old_values_buffer = jnp.empty((new_capacity, num_groups, head_dim), dtype=self.values_buffer.dtype)
            old_values_buffer = old_values_buffer.at[:old_capacity].set(self.values_buffer)
        else:
            old_keys_buffer = self.keys_buffer
            old_values_buffer = self.values_buffer

        dynamic_update_slice_in_dim()
        keys_slice = self.keys_buffer.at[old_sequence_length:new_sequence_length]
        new_keys_buffer = keys_slice.set(new_keys)

        values_slice = self.values_buffer.at[old_sequence_length:new_sequence_length]
        new_values_buffer = values_slice.set(new_values)

        return KVCacheLayer(new_sequence_length, new_keys_buffer, new_values_buffer)

    @classmethod
    def empty(cls, num_groups: int, head_dim: int, dtype: DTypeLike, capacity: int = 0) -> "KVCacheLayer":
        return cls(
            sequence_length=0,
            keys_buffer=jnp.empty((capacity, num_groups, head_dim), dtype=dtype),
            values_buffer=jnp.empty((capacity, num_groups, head_dim), dtype=dtype),
        )
