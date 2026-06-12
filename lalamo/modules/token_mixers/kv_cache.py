from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import dynamic_update_slice_in_dim
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.modules.token_mixer import StateLayerBase
from lalamo.utils.sharding import ShardingConfig

__all__ = ["DynamicKVCacheLayer", "KVCacheLayer", "StaticKVCacheLayer"]


@eqx.filter_jit
def tree_ancestor_mask(
    parent_indices: Int[Array, " nodes"],
) -> Bool[Array, "nodes nodes"]:
    (num_nodes,) = parent_indices.shape
    initial = jnp.eye(num_nodes, dtype=jnp.bool)

    def step(mask: Bool[Array, "nodes nodes"], i: Int[Array, ""]) -> tuple[Bool[Array, "nodes nodes"], None]:
        parent = parent_indices[i]
        parent_row = mask[jnp.maximum(parent, 0)]
        zero_row = jnp.zeros_like(parent_row)
        inherited = jnp.where(parent >= 0, parent_row, zero_row)
        return mask.at[i].set(mask[i] | inherited), None

    mask, _ = jax.lax.scan(step, initial, jnp.arange(num_nodes, dtype=jnp.int32))
    return mask


@eqx.filter_jit
def build_tree_attention_mask(
    total_capacity: int,
    prefix_length: Int[Array, ""] | int,
    parent_indices: Int[Array, " nodes"],
    has_sinks: bool,
) -> Bool[Array, "nodes total_capacity"]:
    prefix_length = jnp.asarray(prefix_length, dtype=jnp.int32)
    (num_nodes,) = parent_indices.shape

    col_indices = jnp.arange(total_capacity, dtype=jnp.int32)
    prefix_mask = col_indices[None, :] < prefix_length

    ancestor_matrix = tree_ancestor_mask(parent_indices)
    draft_offsets = col_indices - prefix_length
    in_draft = (draft_offsets >= 0) & (draft_offsets < num_nodes)
    clamped = jnp.clip(draft_offsets, 0, num_nodes - 1)
    draft_mask = ancestor_matrix[:, clamped] & in_draft[None, :]

    mask = prefix_mask | draft_mask
    if has_sinks:
        mask = mask.at[:, 0].set(True)
    return mask


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
    def padding_mask(self) -> Bool[Array, " tokens"] | None: ...

    @abstractmethod
    def attention_mask(
        self,
        suffix_length: int,
        is_causal: bool,
        suffix_length_without_padding: Int[Array, ""] | int | None = None,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]: ...

    def tree_attention_mask(
        self,
        prefix_length: Int[Array, ""] | int,
        parent_indices: Int[Array, " nodes"],
    ) -> Bool[Array, "nodes tokens"]:
        self._raise_if_batched()
        total, _, _ = self.keys.shape
        mask = build_tree_attention_mask(total, prefix_length, parent_indices, self.has_sinks)
        padding_mask = self.padding_mask
        if padding_mask is not None:
            mask = mask & padding_mask[None, :]
        return mask

    @abstractmethod
    def current_prefix_length(self) -> Int[Array, ""] | int: ...

    @abstractmethod
    def extend(
        self,
        added_keys: Float[Array, "new_tokens groups head_channels"],
        added_values: Float[Array, "new_tokens groups head_channels"],
        added_length: Int[Array, ""] | int | None = None,
    ) -> Self: ...


class DynamicKVCacheLayer(KVCacheLayer):
    padding_mask: Bool[Array, " tokens"] | None = None

    def current_prefix_length(self) -> int:
        self._raise_if_batched()
        return self.keys.shape[0]

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
        return cls(has_sinks=has_sinks, keys=keys, values=values, padding_mask=padding_mask)

    def attention_mask(
        self,
        suffix_length: int,
        is_causal: bool,
        suffix_length_without_padding: Int[Array, ""] | int | None = None,
        sliding_window_size: int | None = None,
    ) -> Bool[Array, "suffix_tokens tokens"]:
        self._raise_if_batched()
        num_tokens, _, _ = self.keys.shape
        if suffix_length_without_padding is None:
            suffix_length_without_padding = suffix_length

        result = jnp.ones((suffix_length, num_tokens), dtype=jnp.bool)
        if is_causal:
            if self.padding_mask is None:
                query_offsets = jnp.arange(0, suffix_length, dtype=jnp.int32) - suffix_length_without_padding
                query_positions = num_tokens + query_offsets
                key_positions = jnp.arange(num_tokens, dtype=jnp.int32)
            else:
                key_positions = jnp.cumsum(self.padding_mask.astype(jnp.int32)) - 1
                query_physical_indices = num_tokens - suffix_length + jnp.arange(suffix_length, dtype=jnp.int32)
                query_positions = key_positions[query_physical_indices]

            result = query_positions[:, None] >= key_positions[None, :]
            if sliding_window_size is not None:
                result = jnp.logical_and(
                    result,
                    query_positions[:, None] < (key_positions[None, :] + sliding_window_size),
                )
        elif sliding_window_size is not None:
            top_zeroed = jnp.tril(result, k=sliding_window_size // 2)
            result = jnp.triu(top_zeroed, k=-sliding_window_size // 2)
        if self.has_sinks:
            result = result.at[:, 0].set(True)
        if self.padding_mask is not None:
            result = jnp.logical_and(result, self.padding_mask[None, :])
        return result

    def extend(
        self,
        added_keys: Float[Array, "new_tokens groups head_channels"],
        added_values: Float[Array, "new_tokens groups head_channels"],
        added_length: Int[Array, ""] | int | None = None,
    ) -> "DynamicKVCacheLayer":
        self._raise_if_batched()
        added_keys = added_keys.astype(self.keys.dtype)
        added_values = added_values.astype(self.values.dtype)
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

    def base_positions(self) -> Int[Array, " batch"]:
        return self.current_length

    def current_prefix_length(self) -> Int[Array, ""]:
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
    def padding_mask(self) -> Bool[Array, " tokens"]:
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

        added_keys = added_keys.astype(self.keys.dtype)
        added_values = added_values.astype(self.values.dtype)
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

    def truncate(self, new_lengths: Int[Array, " batch"]) -> "StaticKVCacheLayer":
        return StaticKVCacheLayer(
            has_sinks=self.has_sinks,
            keys=self.keys,
            values=self.values,
            current_length=new_lengths,
        )

    @classmethod
    def init(
        cls,
        has_sinks: bool,
        capacity: int,
        num_groups: int,
        head_dim: int,
        dtype: DTypeLike,
        sharding_config: ShardingConfig,
    ) -> Self:
        cache_sharding = sharding_config.make_sharding((None, None, None))
        length_sharding = sharding_config.make_sharding(())
        return cls(
            has_sinks=has_sinks,
            keys=jax.device_put(jnp.zeros((capacity, num_groups, head_dim), dtype=dtype), cache_sharding),
            values=jax.device_put(jnp.zeros((capacity, num_groups, head_dim), dtype=dtype), cache_sharding),
            current_length=jax.device_put(jnp.array(has_sinks, dtype=jnp.int32), length_sharding),
        )
