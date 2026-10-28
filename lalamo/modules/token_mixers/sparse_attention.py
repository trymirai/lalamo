from dataclasses import dataclass
from typing import NamedTuple

import jax
from einops import einsum, rearrange
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.kernels.attention import pallas_top_k_blocks, xla_top_k_blocks
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
from .kv_cache import KVCacheLayer

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


def select_top_blocks(
    scores: Float[Array, "queries blocks"],
    num_visible_blocks: Int[Array, " queries"],
    k: int,
    forward_pass_config: MixerForwardPassConfig,
) -> Int[Array, "queries k"]:
    match forward_pass_config.attention_implementation:
        case AttentionImplementation.PALLAS:
            return pallas_top_k_blocks(scores, num_visible_blocks, k)
        case AttentionImplementation.XLA | AttentionImplementation.STABLE_REDUCTION:
            return xla_top_k_blocks(scores, num_visible_blocks, k)


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


class SparseAttentionIndexer(LalamoModule[SparseAttentionIndexerConfig]):
    projection: Linear
    query_norm: Normalization
    key_norm: Normalization

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        state: IndexerCacheLayer | None,
        length_without_padding: Int[Array, ""] | int | None,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> IndexerSelection:
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
        if state is None:
            state = DynamicIndexerCacheLayer.empty(self.config.compression_ratio, self.config.head_dim, raw_keys.dtype)

        ratio = self.config.compression_ratio
        num_suffix_tokens, _ = inputs.shape
        query_positions = state.num_tokens + jnp.arange(num_suffix_tokens, dtype=jnp.int32)
        first_incomplete_blocks = (query_positions + 1) // ratio
        updated_state = state.extend(raw_keys, positional_embeddings, self.key_norm, length_without_padding)
        top_blocks = select_top_blocks(
            block_scores(queries, updated_state.compressed_keys),
            first_incomplete_blocks,
            min(self.config.block_budget, updated_state.num_blocks),
            forward_pass_config,
        )
        selected_blocks = jnp.concatenate([top_blocks, first_incomplete_blocks[:, None]], axis=1)
        positions = rearrange(
            selected_blocks[:, :, None] * ratio + jnp.arange(ratio, dtype=jnp.int32),
            "tokens blocks offsets -> tokens (blocks offsets)",
        )
        is_visible = (positions >= 0) & (positions <= query_positions[:, None])
        return IndexerSelection(
            sparse_mask=SparseMask(positions=jnp.where(is_visible, positions, -1)),
            state=updated_state,
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
        if attention_parent_indices is not None:
            raise NotImplementedError("Sparse attention does not support tree attention")
        if reuse_cache:
            raise NotImplementedError("Sparse attention does not support KV sharing")
        if positional_embeddings is None:
            raise ValueError("Sparse attention requires positional embeddings")
        attention_keychain, indexer_keychain = keychain.split(2)

        selection = self.indexer(
            inputs,
            positional_embeddings,
            state.indexer_cache if state is not None else None,
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
