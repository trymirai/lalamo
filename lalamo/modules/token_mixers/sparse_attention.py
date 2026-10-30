from dataclasses import dataclass
from typing import NamedTuple

import jax
from einops import einsum, rearrange
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.kernels.attention import TOP_K_SCORE_ALIGNMENT, pallas_top_k_blocks, xla_top_k_blocks
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixer import (
    AttentionImplementation,
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

from .attention import Attention, AttentionConfig, SparseMask
from .indexer_cache import DynamicIndexerCacheLayer, IndexerCacheLayer, StaticIndexerCacheLayer
from .kv_cache import KVCacheLayer, tree_ancestor_mask

__all__ = [
    "IndexerSelection",
    "SparseAttention",
    "SparseAttentionConfig",
    "SparseAttentionIndexer",
    "SparseAttentionIndexerConfig",
    "SparseAttentionStateLayer",
]


@dataclass(frozen=True)
class SparseAttentionIndexerConfig(LalamoConfig):
    projection_config: LinearConfig
    query_norm_config: NormalizationConfig
    key_norm_config: NormalizationConfig
    num_heads: int
    head_dim: int
    compression_ratio: int
    token_budget: int

    def __post_init__(self) -> None:
        if self.token_budget % self.compression_ratio != 0:
            raise ValueError(
                f"token_budget ({self.token_budget}) must be divisible by"
                f" compression_ratio ({self.compression_ratio})",
            )

    @property
    def block_budget(self) -> int:
        return self.token_budget // self.compression_ratio

    def init(self, initializer: Initializer, model_dim: int) -> "SparseAttentionIndexer":
        projection = self.projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(self.num_heads * self.head_dim, self.head_dim),
            has_biases=False,
        )
        return SparseAttentionIndexer(
            config=self,
            sharding_config=initializer.sharding_config,
            projection=projection,
            query_norm=self.query_norm_config.init(initializer, input_dim=self.head_dim),
            key_norm=self.key_norm_config.init(initializer, input_dim=self.head_dim),
        )


class IndexerSelection(NamedTuple):
    sparse_mask: SparseMask
    state: IndexerCacheLayer


def block_scores(
    queries: Float[Array, "tokens heads channels"],
    compressed_keys: Float[Array, "blocks channels"],
) -> Float[Array, "tokens blocks"]:
    _, _, head_dim = queries.shape
    head_scores = einsum(
        queries.astype(jnp.float32),
        compressed_keys.astype(jnp.float32),
        "tokens heads channels, blocks channels -> tokens heads blocks",
    )
    return jnp.sum(jax.nn.relu(head_scores), axis=1) * head_dim**-0.5


def path_window_positions_at_ratio(
    suffix_mask: Bool[Array, "suffix_tokens suffix_tokens"],
    path_lengths: Int[Array, " suffix_tokens"],
    num_prefix_tokens: Int[Array, ""],
    compression_ratio: int,
) -> Int[Array, "suffix_tokens ratio"]:
    num_suffix_tokens, _ = suffix_mask.shape
    window_path_lengths = (
        path_lengths[:, None] - (compression_ratio - 1) + jnp.arange(compression_ratio, dtype=jnp.int32)
    )

    suffix_positions = num_prefix_tokens + jnp.arange(num_suffix_tokens, dtype=jnp.int32)
    is_window_suffix_token = suffix_mask[:, :, None] & (path_lengths[None, :, None] == window_path_lengths[:, None, :])
    window_suffix_positions = jnp.max(
        jnp.where(is_window_suffix_token, suffix_positions[None, :, None], -1),
        axis=1,
    )
    window_prefix_positions = window_path_lengths - 1

    return jnp.where(window_path_lengths > num_prefix_tokens, window_suffix_positions, window_prefix_positions)


def candidate_scores(
    queries: Float[Array, "suffix_tokens heads channels"],
    cached_block_keys: Float[Array, "cached_blocks channels"],
    num_complete_cached_blocks: Int[Array, ""],
    suffix_block_keys: Float[Array, "suffix_tokens channels"],
    visible_suffix_blocks: Bool[Array, "suffix_tokens suffix_tokens"],
) -> Float[Array, "suffix_tokens candidate_blocks"]:
    num_cached_blocks, head_dim = cached_block_keys.shape
    num_suffix_tokens, _ = suffix_block_keys.shape
    num_alignment_blocks = -(num_cached_blocks + num_suffix_tokens) % TOP_K_SCORE_ALIGNMENT

    candidate_keys = jnp.concatenate(
        [
            cached_block_keys,
            suffix_block_keys.astype(cached_block_keys.dtype),
            jnp.zeros((num_alignment_blocks, head_dim), dtype=cached_block_keys.dtype),
        ],
    )
    visible_cached_blocks = jnp.broadcast_to(
        jnp.arange(num_cached_blocks) < num_complete_cached_blocks,
        (num_suffix_tokens, num_cached_blocks),
    )
    is_visible = jnp.concatenate(
        [
            visible_cached_blocks,
            visible_suffix_blocks,
            jnp.zeros((num_suffix_tokens, num_alignment_blocks), dtype=jnp.bool),
        ],
        axis=1,
    )
    return jnp.where(is_visible, block_scores(queries, candidate_keys), -jnp.inf)


def select_top_blocks(
    scores: Float[Array, "queries blocks"],
    k: int,
    forward_pass_config: MixerForwardPassConfig,
) -> Int[Array, "queries k"]:
    num_queries, num_blocks = scores.shape
    all_blocks = jnp.full((num_queries,), num_blocks, dtype=jnp.int32)
    match forward_pass_config.attention_implementation:
        case AttentionImplementation.PALLAS:
            top_blocks = pallas_top_k_blocks(scores, all_blocks, k)
        case AttentionImplementation.XLA | AttentionImplementation.STABLE_REDUCTION:
            top_blocks = xla_top_k_blocks(scores, all_blocks, k)
    top_scores = jnp.take_along_axis(scores, jnp.maximum(top_blocks, 0), axis=1)
    return jnp.where((top_blocks >= 0) & jnp.isfinite(top_scores), top_blocks, -1)


def selected_block_positions(
    top_blocks: Int[Array, "suffix_tokens k"],
    num_cached_blocks: int,
    window_positions: Int[Array, "suffix_tokens ratio"],
) -> Int[Array, "suffix_tokens k*ratio"]:
    num_suffix_tokens, compression_ratio = window_positions.shape
    is_suffix_block = top_blocks >= num_cached_blocks
    completing_suffix_tokens = jnp.clip(top_blocks - num_cached_blocks, 0, num_suffix_tokens - 1)
    suffix_block_positions = window_positions[completing_suffix_tokens]
    cached_block_positions = jnp.where(
        top_blocks[:, :, None] >= 0,
        top_blocks[:, :, None] * compression_ratio + jnp.arange(compression_ratio, dtype=jnp.int32),
        -1,
    )
    block_positions = jnp.where(is_suffix_block[:, :, None], suffix_block_positions, cached_block_positions)
    return rearrange(block_positions, "tokens blocks ratio -> tokens (blocks ratio)")


def tail_positions(
    window_positions: Int[Array, "suffix_tokens ratio"],
    tail_lengths: Int[Array, " suffix_tokens"],
) -> Int[Array, "suffix_tokens ratio"]:
    _, compression_ratio = window_positions.shape
    is_tail = jnp.arange(compression_ratio) >= compression_ratio - tail_lengths[:, None]
    return jnp.where(is_tail, window_positions, -1)


class SparseAttentionIndexer(LalamoModule[SparseAttentionIndexerConfig]):
    projection: Linear
    query_norm: Normalization
    key_norm: Normalization

    def project(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "suffix_tokens heads head_channels"], Float[Array, "suffix_tokens head_channels"]]:
        raw_queries, raw_keys = call_vmapped(
            self.projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        query_heads = rearrange(
            raw_queries,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        normalized_query_heads = call_vmapped_twice(self.query_norm, query_heads)
        queries = call_vmapped(positional_embeddings.apply, normalized_query_heads, in_axes=1, out_axes=1)
        return queries, raw_keys

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        state: IndexerCacheLayer | None,
        suffix_mask: Bool[Array, "suffix_tokens suffix_tokens"],
        length_without_padding: Int[Array, ""] | int | None,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> IndexerSelection:
        queries, raw_keys = self.project(inputs, positional_embeddings, forward_pass_config, keychain=keychain)
        if state is None:
            state = DynamicIndexerCacheLayer.empty(self.config.compression_ratio, self.config.head_dim, raw_keys.dtype)
        ratio = self.config.compression_ratio
        num_suffix_tokens, _ = inputs.shape

        path_lengths = state.num_tokens + jnp.sum(suffix_mask, axis=1, dtype=jnp.int32)
        window_positions = path_window_positions_at_ratio(suffix_mask, path_lengths, state.num_tokens, ratio)
        suffix_block_keys = state.compressed_blocks_at(
            raw_keys, positional_embeddings, self.key_norm, window_positions
        )
        completes_block = path_lengths % ratio == 0

        scores = candidate_scores(
            queries,
            state.compressed_keys,
            state.num_tokens // ratio,
            suffix_block_keys,
            suffix_mask & completes_block[None, :],
        )
        top_blocks = select_top_blocks(
            scores,
            min(self.config.block_budget, state.num_blocks + num_suffix_tokens),
            forward_pass_config,
        )

        positions = jnp.concatenate(
            [
                selected_block_positions(top_blocks, state.num_blocks, window_positions),
                tail_positions(window_positions, path_lengths % ratio),
            ],
            axis=1,
        )
        return IndexerSelection(
            sparse_mask=SparseMask(positions=positions),
            state=state.extend(raw_keys, positional_embeddings, self.key_norm, length_without_padding),
        )


@dataclass(frozen=True)
class SparseAttentionConfig(TokenMixerConfig):
    attention_config: AttentionConfig
    indexer_config: SparseAttentionIndexerConfig

    def __post_init__(self) -> None:
        if not self.attention_config.is_causal:
            raise ValueError("Sparse attention requires causal attention")
        if self.attention_config.has_sinks:
            raise ValueError("Sparse attention does not support attention sinks")
        if self.attention_config.sliding_window_size is not None:
            raise ValueError("Sparse attention does not support sliding window attention")
        if self.attention_config.is_kv_sharing:
            raise ValueError("Sparse attention does not support KV sharing")

    def init(self, initializer: Initializer, model_dim: int) -> "SparseAttention":
        return SparseAttention(
            config=self,
            sharding_config=initializer.sharding_config,
            attention=self.attention_config.init(initializer, model_dim),
            indexer=self.indexer_config.init(initializer, model_dim),
        )


class SparseAttentionStateLayer(StateLayerBase):
    kv_cache: KVCacheLayer
    indexer_cache: IndexerCacheLayer


class SparseAttention(TokenMixerBase[SparseAttentionConfig, SparseAttentionStateLayer]):
    attention: Attention
    indexer: SparseAttentionIndexer

    @property
    def model_dim(self) -> int:
        return self.attention.model_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return self.attention.positional_embedding_selector

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: SparseAttentionStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        reuse_cache: bool = False,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[SparseAttentionStateLayer]:
        if reuse_cache:
            raise NotImplementedError("Sparse attention does not support KV sharing")
        if positional_embeddings is None:
            raise ValueError("Sparse attention requires positional embeddings")
        attention_keychain, indexer_keychain = keychain.split(2)

        num_suffix_tokens, _ = inputs.shape
        if attention_parent_indices is None:
            suffix_mask = jnp.tril(jnp.ones((num_suffix_tokens, num_suffix_tokens), dtype=jnp.bool))
        else:
            suffix_mask = tree_ancestor_mask(attention_parent_indices)
        selection = self.indexer(
            inputs,
            positional_embeddings,
            state.indexer_cache if state is not None else None,
            suffix_mask,
            length_without_padding,
            forward_pass_config,
            keychain=indexer_keychain,
        )
        attention_result = self.attention(
            inputs,
            positional_embeddings,
            state.kv_cache if state is not None else None,
            return_updated_state=return_updated_state,
            length_without_padding=length_without_padding,
            forward_pass_config=forward_pass_config,
            keychain=attention_keychain,
            sparse_mask=selection.sparse_mask,
        )

        if attention_result.state is None:
            updated_state = None
        else:
            updated_state = SparseAttentionStateLayer(kv_cache=attention_result.state, indexer_cache=selection.state)
        return TokenMixerResult(outputs=attention_result.outputs, state=updated_state)

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> SparseAttentionStateLayer:
        return SparseAttentionStateLayer(
            kv_cache=self.attention.init_static_state(capacity, dtype),
            indexer_cache=StaticIndexerCacheLayer.init(
                capacity,
                self.config.indexer_config.compression_ratio,
                self.config.indexer_config.head_dim,
                dtype,
                self.sharding_config,
            ),
        )
