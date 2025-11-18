from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.lax import dynamic_update_slice_in_dim
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.common import ParameterTree

from .common import StateLayerBase

__all__ = ["DynamicKVCacheLayer", "KVCacheLayer", "StaticKVCacheLayer"]


class KVCacheLayer(StateLayerBase):
    has_sinks: bool = eqx.field(static=True)
    keys: Float[Array, "*batch tokens groups head_channels"]
    values: Float[Array, "*batch tokens groups head_channels"]

    def __post_init__(self) -> None:
        if self.keys.ndim not in (3, 4):
            raise ValueError(
                f"Key and value buffers must have 3 or 4 dimensions: [batch], capacity, groups, head_channels,"
                f" got shape {self.keys.shape}",
            )
        if self.keys.shape != self.values.shape:
            raise ValueError("Keys and values buffers must have the same shape")
        if self.keys.dtype != self.values.dtype:
            raise ValueError("Keys and values buffers must have the same dtype")

    def _raise_if_batched(self) -> None:
        if self.keys.ndim != 3:
            raise ValueError(
                "Attempted to call a method on a batched version of KVCacheLayer. Use vmap instead.",
            )

    @abstractmethod
    def attention_mask(
        self,
        suffix_length: int,
        is_causal: bool,
        suffix_length_without_padding: Int[Array, ""] | int | None = None,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]: ...

    @abstractmethod
    def extend(
        self,
        added_keys: Float[Array, "new_tokens groups head_channels"],
        added_values: Float[Array, "new_tokens groups head_channels"],
        added_length: Int[Array, ""] | int | None = None,
    ) -> Self: ...

    def export(self) -> ParameterTree:
        return dict(
            keys=self.keys,
            values=self.values,
        )


class DynamicKVCacheLayer(KVCacheLayer):
    padding_mask: Bool[Array, " tokens"] | None = None

    @classmethod
    def init(
        cls,
        has_sinks: bool,
        keys: Float[Array, "tokens groups head_channels"],
        values: Float[Array, "tokens groups head_channels"],
        length: Int[Array, ""] | int | None = None,
    ) -> "DynamicKVCacheLayer":
        num_tokens, num_groups, head_dim = keys.shape
        if length is None:
            padding_mask = None
        else:
            token_indices = jnp.arange(num_tokens, dtype=jnp.int32)
            padding_mask = token_indices < length
        if has_sinks:
            sinks = jnp.zeros((1, num_groups, head_dim), dtype=keys.dtype)
            keys = jnp.concatenate([sinks, keys], axis=0)
            values = jnp.concatenate([sinks, values], axis=0)
            if padding_mask is not None:
                true = jnp.ones((1,), dtype=jnp.bool)
                padding_mask = jnp.concatenate([true, padding_mask], axis=0)
        return cls(has_sinks, keys, values, padding_mask)

    def attention_mask(
        self,
        suffix_length: int,
        is_causal: bool,
        suffix_length_without_padding: (Int[Array, ""] | int | None) = None,  # noqa: ARG002
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]:
        self._raise_if_batched()
        total_num_tokens, _, _ = self.keys.shape
        result = jnp.ones((suffix_length, total_num_tokens), dtype=jnp.bool)
        if is_causal:
            result = jnp.tril(result, k=total_num_tokens - suffix_length)
            if sliding_window_size is not None:
                result = jnp.triu(result, k=1 - sliding_window_size)
        elif sliding_window_size is not None:
            top_zeroed = jnp.tril(result, k=sliding_window_size // 2)
            result = jnp.triu(top_zeroed, k=-sliding_window_size // 2)
        if self.has_sinks:
            result = result.at[:, 0].set(True)
        if self.padding_mask is not None:
            result = result & self.padding_mask[None, :]
        return result

    def extend(
        self,
        added_keys: Float[Array, "new_tokens groups head_channels"],
        added_values: Float[Array, "new_tokens groups head_channels"],
        added_length: Int[Array, ""] | int | None = None,
    ) -> "DynamicKVCacheLayer":
        self._raise_if_batched()
        updated_keys = jnp.concatenate([self.keys, added_keys], axis=0)
        updated_values = jnp.concatenate([self.values, added_values], axis=0)

        added_padded_length, _, _ = added_keys.shape
        if self.padding_mask is None and added_length is None:
            return DynamicKVCacheLayer(self.has_sinks, updated_keys, updated_values)
        if added_length is None:
            added_length = added_padded_length

        if self.padding_mask is not None:
            old_padding_mask = self.padding_mask
        else:
            old_num_tokens, _, _ = self.keys.shape
            old_padding_mask = jnp.ones(old_num_tokens, dtype=jnp.bool)

        added_padding_mask = jnp.arange(added_padded_length, dtype=jnp.int32) < added_length
        updated_padding_mask = jnp.concatenate([old_padding_mask, added_padding_mask], axis=0)
        return DynamicKVCacheLayer(self.has_sinks, updated_keys, updated_values, updated_padding_mask)


class StaticKVCacheLayer(KVCacheLayer):
    current_length: Int[Array, "*batch"]

    def attention_mask(
        self,
        suffix_length: int,
        is_causal: bool,
        suffix_length_without_padding: Int[Array, ""] | int | None = None,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]:
        self._raise_if_batched()
        if suffix_length_without_padding is None:
            suffix_length_without_padding = suffix_length
        if is_causal:
            query_offsets = jnp.arange(0, suffix_length, dtype=jnp.int32) - suffix_length_without_padding
        else:
            query_offsets = jnp.zeros(suffix_length, dtype=jnp.int32)

        query_indices = self.current_length + query_offsets
        key_indices = jnp.arange(self.capacity, dtype=jnp.int32)

        result = query_indices[:, None] >= key_indices[None, :]
        if sliding_window_size is not None:
            swa_mask = query_indices[:, None] < (key_indices[None, :] + sliding_window_size)
            result = result & swa_mask
        if self.has_sinks:
            result = result.at[:, 0].set(True)

        return result

    @property
    def padding_mask(self) -> Bool[Array, " tokens"] | None:
        self._raise_if_batched()
        return jnp.arange(self.capacity, dtype=jnp.int32) < self.current_length

    @property
    def capacity(self) -> int:
        self._raise_if_batched()
        result, _, _ = self.keys.shape
        return result

    def extend(
        self,
        added_keys: Float[Array, "tokens groups head_channels"],
        added_values: Float[Array, "tokens groups head_channels"],
        added_length: Int[Array, ""] | int | None = None,
    ) -> "StaticKVCacheLayer":
        self._raise_if_batched()
        if added_keys.shape != added_values.shape:
            raise ValueError("Keys and values must have the same shape")
        num_added_tokens, new_num_groups, new_head_dim = added_keys.shape
        _, old_num_groups, old_head_dim = self.keys.shape
        if new_num_groups != old_num_groups or new_head_dim != old_head_dim:
            raise ValueError("New keys and values must have the same number of groups and head dimensions")

        if added_length is None:
            added_length = num_added_tokens

        updated_keys = dynamic_update_slice_in_dim(
            self.keys,
            added_keys,
            self.current_length,
            0,
            allow_negative_indices=False,
        )
        updated_values = dynamic_update_slice_in_dim(
            self.values,
            added_values,
            self.current_length,
            0,
            allow_negative_indices=False,
        )
        updated_sequence_length = self.current_length + added_length
        return StaticKVCacheLayer(
            has_sinks=self.has_sinks,
            keys=updated_keys,
            values=updated_values,
            current_length=updated_sequence_length,
        )

    @classmethod
    def init(
        cls,
        has_sinks: bool,
        capacity: int,
        num_groups: int,
        head_dim: int,
        dtype: DTypeLike,
    ) -> Self:
        return cls(
            has_sinks=has_sinks,
            keys=jnp.zeros((capacity, num_groups, head_dim), dtype=dtype),
            values=jnp.zeros((capacity, num_groups, head_dim), dtype=dtype),
            current_length=jnp.array(has_sinks, dtype=jnp.int32),
        )
