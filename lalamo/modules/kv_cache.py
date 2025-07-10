from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.lax import dynamic_update_slice_in_dim
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.common import ParameterDict

__all__ = ["DynamicKVCacheLayer", "KVCacheLayer", "StaticKVCacheLayer"]


class KVCacheLayer(eqx.Module):
    keys: Float[Array, "tokens groups head_channels"]
    values: Float[Array, "tokens groups head_channels"]

    def __post_init__(self) -> None:
        if self.keys.ndim != 3:
            raise ValueError(
                f"Key and value buffers must have 3 dimensions: capacity, groups, head_channels,"
                f" got shape {self.keys.shape}",
            )
        if self.keys.shape != self.values.shape:
            raise ValueError("Keys and values buffers must have the same shape")
        if self.keys.dtype != self.values.dtype:
            raise ValueError("Keys and values buffers must have the same dtype")

    @abstractmethod
    def attention_mask(
        self,
        suffix_length: int,
        is_causal: bool,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, ""]: ...

    @abstractmethod
    def extend(
        self,
        new_keys: Float[Array, "new_tokens groups head_channels"],
        new_values: Float[Array, "new_tokens groups head_channels"],
    ) -> Self: ...

    def export(self) -> ParameterDict:
        return ParameterDict(
            keys=self.keys,
            values=self.values,
        )


class DynamicKVCacheLayer(KVCacheLayer):
    def attention_mask(
        self,
        suffix_length: int,
        is_causal: bool,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, ""]:
        total_num_tokens, _, _ = self.keys.shape
        result = jnp.ones((suffix_length, total_num_tokens), dtype=jnp.bool)
        if is_causal:
            result = jnp.tril(result, k=total_num_tokens - suffix_length)
        if sliding_window_size is not None:
            result = jnp.triu(result, k=1 - sliding_window_size)
        return result

    def extend(
        self,
        new_keys: Float[Array, "new_tokens groups head_channels"],
        new_values: Float[Array, "new_tokens groups head_channels"],
    ) -> "DynamicKVCacheLayer":
        updated_keys = jnp.concatenate([self.keys, new_keys], axis=0)
        updated_values = jnp.concatenate([self.values, new_values], axis=0)
        return DynamicKVCacheLayer(updated_keys, updated_values)


class StaticKVCacheLayer(KVCacheLayer):
    current_length: Int[Array, ""]

    def attention_mask(
        self,
        suffix_length: int,
        is_causal: bool,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, ""]:
        if is_causal:
            query_offsets = jnp.arange(-suffix_length, 0, dtype=jnp.int32)
        else:
            query_offsets = jnp.zeros(suffix_length, dtype=jnp.int32)

        query_indices = self.current_length + query_offsets
        key_indices = jnp.arange(self.capacity, dtype=jnp.int32)

        result = query_indices[:, None] >= key_indices[None, :]
        if sliding_window_size is not None:
            swa_mask = query_indices[:, None] < (key_indices[None, :] + sliding_window_size)
            result = result & swa_mask

        return result

    @property
    def padding_mask(self) -> Bool[Array, ""] | None:
        return jnp.arange(self.capacity, dtype=jnp.int32) < self.current_length

    @property
    def capacity(self) -> int:
        result, _, _ = self.keys.shape
        return result

    def extend(
        self,
        new_keys: Float[Array, "tokens groups head_channels"],
        new_values: Float[Array, "tokens groups head_channels"],
    ) -> "StaticKVCacheLayer":
        if new_keys.shape != new_values.shape:
            raise ValueError("Keys and values must have the same shape")
        num_new_tokens, new_num_groups, new_head_dim = new_keys.shape
        _, old_num_groups, old_head_dim = self.keys.shape
        if new_num_groups != old_num_groups or new_head_dim != old_head_dim:
            raise ValueError("New keys and values must have the same number of groups and head dimensions")

        updated_keys = dynamic_update_slice_in_dim(
            self.keys,
            new_keys,
            self.current_length,
            0,
            allow_negative_indices=False,
        )
        updated_values = dynamic_update_slice_in_dim(
            self.values,
            new_values,
            self.current_length,
            0,
            allow_negative_indices=False,
        )
        updated_sequence_length = self.current_length + num_new_tokens
        return StaticKVCacheLayer(keys=updated_keys, values=updated_values, current_length=updated_sequence_length)

    @classmethod
    def empty(cls, capacity: int, num_groups: int, head_dim: int, dtype: DTypeLike) -> Self:
        return cls(
            keys=jnp.empty((capacity, num_groups, head_dim), dtype=dtype),
            values=jnp.empty((capacity, num_groups, head_dim), dtype=dtype),
            current_length=jnp.array(0, dtype=jnp.int32),
        )
