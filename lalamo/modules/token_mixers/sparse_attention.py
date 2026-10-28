from dataclasses import dataclass
from typing import NamedTuple

import jax
from einops import einsum, rearrange
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.kernels.attention import (
    pallas_sparse_attention,
    pallas_top_k_blocks,
    sparse_rows_mask,
    stable_reduction_attention,
    xla_sparse_attention,
    xla_top_k_blocks,
)
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

from .indexer_cache import DynamicIndexerCacheLayer, IndexerCacheLayer, StaticIndexerCacheLayer
from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, StaticKVCacheLayer

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
    selected_blocks: Int[Array, "suffix_tokens selected_blocks"]
    query_positions: Int[Array, " suffix_tokens"]
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


def prepare_heads(
    projection: Float[Array, "tokens channels"],
    num_heads: int,
    head_dim: int,
    norm: Normalization,
    positional_embeddings: PositionalEmbeddings,
) -> Float[Array, "tokens heads head_channels"]:
    heads = rearrange(
        projection,
        "tokens (heads head_channels) -> tokens heads head_channels",
        heads=num_heads,
        head_channels=head_dim,
    )
    heads = call_vmapped_twice(norm, heads)
    return call_vmapped(positional_embeddings.apply, heads, in_axes=1, out_axes=1)


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
        queries = prepare_heads(
            raw_queries,
            self.config.num_heads,
            self.config.head_dim,
            self.query_norm,
            positional_embeddings,
        )
        if state is None:
            state = DynamicIndexerCacheLayer.empty(self.config.compression_ratio, self.config.head_dim, raw_keys.dtype)

        num_suffix_tokens, _ = inputs.shape
        query_positions = state.num_tokens + jnp.arange(num_suffix_tokens, dtype=jnp.int32)
        first_incomplete_blocks = (query_positions + 1) // self.config.compression_ratio
        updated_state = state.extend(raw_keys, positional_embeddings, self.key_norm, length_without_padding)
        top_blocks = select_top_blocks(
            block_scores(queries, updated_state.compressed_keys),
            first_incomplete_blocks,
            min(self.config.block_budget, updated_state.num_blocks),
            forward_pass_config,
        )
        return IndexerSelection(
            selected_blocks=jnp.concatenate([top_blocks, first_incomplete_blocks[:, None]], axis=1),
            query_positions=query_positions,
            state=updated_state,
        )


@dataclass(frozen=True)
class SparseAttentionConfig(TokenMixerConfig):
    qkvg_projection_config: LinearConfig
    out_projection_config: LinearConfig
    query_norm_config: NormalizationConfig
    key_norm_config: NormalizationConfig
    indexer_config: SparseAttentionIndexerConfig
    num_heads: int
    num_groups: int
    head_dim: int

    def init(self, initializer: Initializer, model_dim: int) -> "SparseAttention":
        query_dim = self.num_heads * self.head_dim
        key_value_dim = self.num_groups * self.head_dim
        return SparseAttention(
            config=self,
            sharding_config=initializer.sharding_config,
            qkvg_projection=self.qkvg_projection_config.init(
                initializer,
                input_dim=model_dim,
                output_dims=(query_dim, key_value_dim, key_value_dim, query_dim),
                has_biases=False,
            ),
            out_projection=self.out_projection_config.init(
                initializer,
                query_dim,
                (model_dim,),
                has_biases=False,
            ),
            query_norm=self.query_norm_config.init(initializer, input_dim=self.head_dim),
            key_norm=self.key_norm_config.init(initializer, input_dim=self.head_dim),
            indexer=self.indexer_config.init(initializer, model_dim),
        )


class SparseAttentionStateLayer(StateLayerBase):
    kv_cache: KVCacheLayer
    indexer_cache: IndexerCacheLayer


def selected_rows(
    kv_cache: KVCacheLayer,
    selection: IndexerSelection,
    compression_ratio: int,
) -> Int[Array, "suffix_tokens rows"]:
    num_slots, _, _ = kv_cache.keys.shape
    positions = rearrange(
        selection.selected_blocks[:, :, None] * compression_ratio + jnp.arange(compression_ratio, dtype=jnp.int32),
        "tokens blocks offsets -> tokens (blocks offsets)",
    )
    is_valid = (positions >= 0) & (positions <= selection.query_positions[:, None])
    return jnp.where(is_valid, kv_cache.slots_for_positions(positions), num_slots)


def sparse_attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    *,
    rows: Int[Array, "dst_tokens rows"],
    forward_pass_config: MixerForwardPassConfig,
) -> Float[Array, "dst_tokens heads head_channels"]:
    num_src_tokens, _, _ = keys.shape
    match forward_pass_config.attention_implementation:
        case AttentionImplementation.PALLAS:
            return pallas_sparse_attention(queries, keys, values, rows, None, None)
        case AttentionImplementation.XLA:
            return xla_sparse_attention(queries, keys, values, rows, None, None)
        case AttentionImplementation.STABLE_REDUCTION:
            return stable_reduction_attention(
                queries,
                keys,
                values,
                bias=None,
                mask=sparse_rows_mask(rows, num_src_tokens),
                scale=None,
                logit_soft_cap=None,
                tile_size=forward_pass_config.attention_tile_size,
                accumulation_dtype=forward_pass_config.attention_accumulation_dtype,
            )


class SparseAttention(TokenMixerBase[SparseAttentionConfig, SparseAttentionStateLayer]):
    qkvg_projection: Linear
    out_projection: Linear
    query_norm: Normalization
    key_norm: Normalization
    indexer: SparseAttentionIndexer

    @property
    def model_dim(self) -> int:
        return self.qkvg_projection.input_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return PositionalEmbeddingSelector.GLOBAL

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
        qkvg_keychain, indexer_keychain, out_keychain = keychain.split(3)

        raw_queries, raw_keys, raw_values, gate = call_vmapped(
            self.qkvg_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=qkvg_keychain,
        )
        queries = prepare_heads(
            raw_queries,
            self.config.num_heads,
            self.config.head_dim,
            self.query_norm,
            positional_embeddings,
        )
        keys = prepare_heads(
            raw_keys,
            self.config.num_groups,
            self.config.head_dim,
            self.key_norm,
            positional_embeddings,
        )
        values = rearrange(
            raw_values,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_groups,
            head_channels=self.config.head_dim,
        )

        if state is None:
            kv_cache = DynamicKVCacheLayer.init(
                has_sinks=False,
                keys=keys.astype(values.dtype),
                values=values,
                length=length_without_padding,
            )
            indexer_cache = None
        else:
            kv_cache = state.kv_cache.extend(keys, values, added_length=length_without_padding)
            indexer_cache = state.indexer_cache

        selection = self.indexer(
            inputs,
            positional_embeddings,
            indexer_cache,
            length_without_padding,
            forward_pass_config,
            keychain=indexer_keychain,
        )
        attention_output = sparse_attention_kernel(
            queries.astype(kv_cache.keys.dtype),
            kv_cache.keys,
            kv_cache.values,
            rows=selected_rows(kv_cache, selection, self.config.indexer_config.compression_ratio),
            forward_pass_config=forward_pass_config,
        )
        gated_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
        ) * jax.nn.sigmoid(gate)
        (result,) = call_vmapped(
            self.out_projection,
            gated_output,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain,
        )

        if return_updated_state:
            updated_state = SparseAttentionStateLayer(kv_cache=kv_cache, indexer_cache=selection.state)
        else:
            updated_state = None
        return TokenMixerResult(outputs=result, state=updated_state)

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> SparseAttentionStateLayer:
        return SparseAttentionStateLayer(
            kv_cache=StaticKVCacheLayer.init(
                has_sinks=False,
                capacity=capacity,
                num_groups=self.config.num_groups,
                head_dim=self.config.head_dim,
                dtype=dtype,
                sharding_config=self.sharding_config,
            ),
            indexer_cache=StaticIndexerCacheLayer.init(
                capacity,
                self.config.indexer_config.compression_ratio,
                self.config.indexer_config.head_dim,
                dtype,
                self.sharding_config,
            ),
        )
