import math
from abc import abstractmethod
from dataclasses import replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.lax import dynamic_slice_in_dim
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.modules.normalization import Normalization
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixer import StateLayerBase
from lalamo.modules.utils import call_vmapped
from lalamo.utils.sharding import ShardingConfig

__all__ = ["DynamicIndexerCacheLayer", "IndexerCacheLayer", "StaticIndexerCacheLayer"]


class IndexerCacheLayer(StateLayerBase):
    compression_ratio: int = eqx.field(static=True)
    compressed_keys: Float[Array, "*batch blocks head_channels"]
    tail_keys: Float[Array, "*batch tail_tokens head_channels"]
    tail_cosines: Float[Array, "*batch tail_tokens head_channels"]
    tail_sines: Float[Array, "*batch tail_tokens head_channels"]
    num_tokens: Int[Array, "*batch"]

    def __post_init__(self) -> None:
        if self.compressed_keys.ndim not in (2, 3):
            raise ValueError(
                "Compressed keys must have 2 or 3 dimensions: [batch], blocks, head_channels,"
                f" got shape {self.compressed_keys.shape}",
            )
        *_, num_tail_tokens, _ = self.tail_keys.shape
        if num_tail_tokens != self.compression_ratio - 1:
            raise ValueError(
                f"Tail must hold compression_ratio - 1 = {self.compression_ratio - 1} tokens, got {num_tail_tokens}",
            )

    def raise_if_batched(self) -> None:
        if self.compressed_keys.ndim != 2:
            raise ValueError(
                "Attempted to call a method on a batched version of IndexerCacheLayer. Use vmap instead.",
            )

    @property
    def num_blocks(self) -> int:
        self.raise_if_batched()
        num_blocks, _ = self.compressed_keys.shape
        return num_blocks

    @abstractmethod
    def compressed_keys_with_room_for(self, num_added_tokens: int) -> Float[Array, "blocks head_channels"]: ...

    def extend(
        self,
        added_keys: Float[Array, "new_tokens head_channels"],
        positional_embeddings: PositionalEmbeddings,
        key_norm: Normalization,
        added_length: Int[Array, ""] | int | None = None,
    ) -> Self:
        self.raise_if_batched()
        num_new_tokens, _ = added_keys.shape
        if added_length is None:
            added_length = num_new_tokens
        ratio = self.compression_ratio
        rope_dim = positional_embeddings.head_dim
        rotation_dtype = positional_embeddings.cosines.dtype
        tail_length = self.num_tokens % ratio
        num_touched_blocks = math.ceil((ratio - 1 + num_new_tokens) / ratio)
        histories = (
            jnp.concatenate([self.tail_keys, added_keys.astype(self.tail_keys.dtype)]),
            jnp.concatenate([self.tail_cosines[:, :rope_dim].astype(rotation_dtype), positional_embeddings.cosines]),
            jnp.concatenate([self.tail_sines[:, :rope_dim].astype(rotation_dtype), positional_embeddings.sines]),
        )
        block_keys, block_cosines, block_sines = (
            dynamic_slice_in_dim(
                jnp.pad(history, ((0, num_touched_blocks * ratio - num_new_tokens), (0, 0))),
                ratio - 1 - tail_length,
                num_touched_blocks * ratio,
            )
            for history in histories
        )

        grouped_keys = rearrange(
            block_keys,
            "(blocks ratio) head_channels -> blocks ratio head_channels",
            ratio=ratio,
        )
        pooled_keys = jnp.mean(grouped_keys.astype(jnp.float32), axis=1).astype(grouped_keys.dtype)
        block_start_embeddings = PositionalEmbeddings(cosines=block_cosines[::ratio], sines=block_sines[::ratio])
        rotated_keys = block_start_embeddings.apply(call_vmapped(key_norm, pooled_keys))

        compressed_keys = self.compressed_keys_with_room_for(num_new_tokens)
        num_capacity_blocks, _ = compressed_keys.shape
        num_completed_blocks = (tail_length + added_length) // ratio
        block_offsets = jnp.arange(num_touched_blocks, dtype=jnp.int32)
        target_blocks = jnp.where(
            block_offsets < num_completed_blocks,
            self.num_tokens // ratio + block_offsets,
            num_capacity_blocks,
        )
        compressed_keys = compressed_keys.at[target_blocks].set(
            rotated_keys.astype(compressed_keys.dtype),
            mode="drop",
        )

        tail_keys, tail_cosines, tail_sines = (
            dynamic_slice_in_dim(history, added_length, ratio - 1) for history in histories
        )
        return replace(
            self,
            compressed_keys=compressed_keys,
            tail_keys=tail_keys,
            tail_cosines=self.tail_cosines.at[:, :rope_dim].set(tail_cosines.astype(self.tail_cosines.dtype)),
            tail_sines=self.tail_sines.at[:, :rope_dim].set(tail_sines.astype(self.tail_sines.dtype)),
            num_tokens=self.num_tokens + added_length,
        )


class DynamicIndexerCacheLayer(IndexerCacheLayer):
    def compressed_keys_with_room_for(self, num_added_tokens: int) -> Float[Array, "blocks head_channels"]:
        _, head_dim = self.compressed_keys.shape
        num_added_blocks = math.ceil(num_added_tokens / self.compression_ratio)
        padding = jnp.zeros((num_added_blocks, head_dim), dtype=self.compressed_keys.dtype)
        return jnp.concatenate([self.compressed_keys, padding], axis=0)

    @classmethod
    def empty(cls, compression_ratio: int, head_dim: int, dtype: DTypeLike) -> Self:
        return cls(
            compression_ratio=compression_ratio,
            compressed_keys=jnp.zeros((0, head_dim), dtype=dtype),
            tail_keys=jnp.zeros((compression_ratio - 1, head_dim), dtype=dtype),
            tail_cosines=jnp.zeros((compression_ratio - 1, head_dim), dtype=jnp.float32),
            tail_sines=jnp.zeros((compression_ratio - 1, head_dim), dtype=jnp.float32),
            num_tokens=jnp.array(0, dtype=jnp.int32),
        )


class StaticIndexerCacheLayer(IndexerCacheLayer):
    def compressed_keys_with_room_for(self, num_added_tokens: int) -> Float[Array, "blocks head_channels"]:  # noqa: ARG002
        return self.compressed_keys

    @classmethod
    def init(
        cls,
        capacity: int,
        compression_ratio: int,
        head_dim: int,
        dtype: DTypeLike,
        sharding_config: ShardingConfig,
    ) -> Self:
        matrix_sharding = sharding_config.make_sharding((None, None))
        scalar_sharding = sharding_config.make_sharding(())
        num_blocks = math.ceil(capacity / compression_ratio)
        tail_shape = (compression_ratio - 1, head_dim)
        return cls(
            compression_ratio=compression_ratio,
            compressed_keys=jax.device_put(jnp.zeros((num_blocks, head_dim), dtype=dtype), matrix_sharding),
            tail_keys=jax.device_put(jnp.zeros(tail_shape, dtype=dtype), matrix_sharding),
            tail_cosines=jax.device_put(jnp.zeros(tail_shape, dtype=jnp.float32), matrix_sharding),
            tail_sines=jax.device_put(jnp.zeros(tail_shape, dtype=jnp.float32), matrix_sharding),
            num_tokens=jax.device_put(jnp.array(0, dtype=jnp.int32), scalar_sharding),
        )
