from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.lax import dynamic_update_slice_in_dim
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.common import ParameterTree

from .common import StateLayerBase

__all__ = ["DynamicKVCacheLayer", "KVCacheLayer", "StaticKVCacheLayer"]


def _tree_positions(
    prefix_length: Int[Array, ""] | int,
    parent_indices: Int[Array, " suffix_tokens"],
) -> Int[Array, " suffix_tokens"]:
    prefix_length = jnp.asarray(prefix_length, dtype=jnp.int32)
    positions = jnp.zeros(parent_indices.shape, dtype=jnp.int32)
    for node_offset in range(parent_indices.shape[0]):
        parent_index = parent_indices[node_offset]
        parent_is_tree_node = parent_index >= prefix_length
        parent_offset = jnp.maximum(parent_index - prefix_length, 0)
        parent_position = jnp.where(parent_is_tree_node, positions[parent_offset], parent_index)
        positions = positions.at[node_offset].set(parent_position + 1)
    return positions


def _tree_ancestor_mask(
    prefix_length: Int[Array, ""] | int,
    parent_indices: Int[Array, " suffix_tokens"],
) -> Bool[Array, "suffix_tokens suffix_tokens"]:
    prefix_length = jnp.asarray(prefix_length, dtype=jnp.int32)
    suffix_length = parent_indices.shape[0]
    ancestors = jnp.eye(suffix_length, dtype=jnp.bool)
    empty_row = jnp.zeros((suffix_length,), dtype=jnp.bool)
    for node_offset in range(suffix_length):
        parent_index = parent_indices[node_offset]
        parent_is_tree_node = parent_index >= prefix_length
        parent_offset = jnp.maximum(parent_index - prefix_length, 0)
        parent_ancestors = jnp.where(parent_is_tree_node, ancestors[parent_offset], empty_row)
        ancestors = ancestors.at[node_offset].set(ancestors[node_offset] | parent_ancestors)
    return ancestors


def _build_tree_attention_mask(
    total_num_tokens: int,
    prefix_length: Int[Array, ""] | int,
    parent_indices: Int[Array, " suffix_tokens"],
    has_sinks: bool,
    suffix_length_without_padding: Int[Array, ""] | int | None,
    sliding_window_size: int | None,
) -> Bool[Array, "suffix_tokens tokens"]:
    prefix_length = jnp.asarray(prefix_length, dtype=jnp.int32)
    suffix_length = parent_indices.shape[0]
    token_indices = jnp.arange(total_num_tokens, dtype=jnp.int32)
    prefix_mask = token_indices < prefix_length

    result = jnp.broadcast_to(prefix_mask[None, :], (suffix_length, total_num_tokens))

    suffix_slot_indices = prefix_length + jnp.arange(suffix_length, dtype=jnp.int32)
    tree_mask = _tree_ancestor_mask(prefix_length, parent_indices)
    result = result.at[:, suffix_slot_indices].set(tree_mask)

    if sliding_window_size is not None:
        tree_positions = _tree_positions(prefix_length, parent_indices)
        key_positions = token_indices
        key_positions = key_positions.at[suffix_slot_indices].set(tree_positions)
        within_window = tree_positions[:, None] < (key_positions[None, :] + sliding_window_size)
        result = result & within_window

    if has_sinks:
        result = result.at[:, 0].set(True)

    if suffix_length_without_padding is not None:
        suffix_length_without_padding = jnp.asarray(suffix_length_without_padding, dtype=jnp.int32)
        query_is_valid = jnp.arange(suffix_length, dtype=jnp.int32) < suffix_length_without_padding
        fallback_row = jnp.zeros((total_num_tokens,), dtype=jnp.bool)
        fallback_index = jnp.maximum(prefix_length - 1, 0)
        fallback_row = fallback_row.at[fallback_index].set(True)
        result = jnp.where(query_is_valid[:, None], result, fallback_row[None, :])

    return result


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

    @property
    @abstractmethod
    def length(self) -> Int[Array, ""] | int: ...

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

    @abstractmethod
    def tree_attention_mask(
        self,
        prefix_length: Int[Array, ""] | int,
        parent_indices: Int[Array, " suffix_tokens"],
        suffix_length_without_padding: Int[Array, ""] | int | None = None,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]: ...

    def export(self) -> ParameterTree:
        return dict(
            keys=self.keys,
            values=self.values,
        )


class DynamicKVCacheLayer(KVCacheLayer):
    padding_mask: Bool[Array, " tokens"] | None = None

    @property
    def length(self) -> Int[Array, ""] | int:
        self._raise_if_batched()
        if self.padding_mask is None:
            return self.keys.shape[0]
        return jnp.sum(self.padding_mask, dtype=jnp.int32)

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
        suffix_length_without_padding: Int[Array, ""] | int | None = None,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]:
        self._raise_if_batched()
        total_num_tokens, _, _ = self.keys.shape
        if suffix_length_without_padding is None:
            suffix_length_without_padding = suffix_length

        result = jnp.ones((suffix_length, total_num_tokens), dtype=jnp.bool)
        if is_causal:
            query_offsets = jnp.arange(0, suffix_length, dtype=jnp.int32) - suffix_length_without_padding
            query_indices = self.length + query_offsets
            key_indices = jnp.arange(total_num_tokens, dtype=jnp.int32)

            result = query_indices[:, None] >= key_indices[None, :]
            if sliding_window_size is not None:
                result = result & (query_indices[:, None] < (key_indices[None, :] + sliding_window_size))
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

    def tree_attention_mask(
        self,
        prefix_length: Int[Array, ""] | int,
        parent_indices: Int[Array, " suffix_tokens"],
        suffix_length_without_padding: Int[Array, ""] | int | None = None,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]:
        self._raise_if_batched()
        result = _build_tree_attention_mask(
            total_num_tokens=self.keys.shape[0],
            prefix_length=prefix_length,
            parent_indices=parent_indices,
            has_sinks=self.has_sinks,
            suffix_length_without_padding=suffix_length_without_padding,
            sliding_window_size=sliding_window_size,
        )
        if self.padding_mask is None:
            return result
        return result & self.padding_mask[None, :]


class StaticKVCacheLayer(KVCacheLayer):
    current_length: Int[Array, "*batch"]

    @property
    def length(self) -> Int[Array, ""] | int:
        self._raise_if_batched()
        return self.current_length

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

    def tree_attention_mask(
        self,
        prefix_length: Int[Array, ""] | int,
        parent_indices: Int[Array, " suffix_tokens"],
        suffix_length_without_padding: Int[Array, ""] | int | None = None,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]:
        self._raise_if_batched()
        return _build_tree_attention_mask(
            total_num_tokens=self.capacity,
            prefix_length=prefix_length,
            parent_indices=parent_indices,
            has_sinks=self.has_sinks,
            suffix_length_without_padding=suffix_length_without_padding,
            sliding_window_size=sliding_window_size,
        ) & self.padding_mask[None, :]

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
