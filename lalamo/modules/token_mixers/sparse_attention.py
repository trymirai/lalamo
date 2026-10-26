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
    StateLayerBase,
    TokenMixerResult,
)
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

from .attention import AttentionBase, AttentionConfigBase, SharedKeyValueProjection
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
        compression_ratio = self.config.compression_ratio
        raw_queries, raw_keys = call_vmapped(
            self.projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        queries = rearrange(
            raw_queries,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        queries = call_vmapped_twice(self.query_norm, queries)
        queries = call_vmapped(positional_embeddings.apply, queries, in_axes=1, out_axes=1)

        if state is None:
            state = DynamicIndexerCacheLayer.empty(compression_ratio, self.config.head_dim, raw_keys.dtype)
        updated_state = state.extend(raw_keys, positional_embeddings, self.key_norm, length_without_padding)

        num_suffix_tokens, _, _ = queries.shape
        query_positions = state.num_tokens + jnp.arange(num_suffix_tokens, dtype=jnp.int32)
        num_visible_blocks = (query_positions + 1) // compression_ratio
        head_scores = einsum(
            queries.astype(jnp.float32),
            updated_state.compressed_keys.astype(jnp.float32),
            "tokens heads channels, blocks channels -> tokens heads blocks",
        )
        scores = jnp.sum(jax.nn.relu(head_scores), axis=1) * self.config.head_dim**-0.5
        selected_blocks = select_top_blocks(
            scores,
            num_visible_blocks,
            min(self.config.block_budget, updated_state.num_blocks),
            forward_pass_config,
        )
        tail_blocks = jnp.where((query_positions + 1) % compression_ratio != 0, num_visible_blocks, -1)
        return IndexerSelection(
            selected_blocks=jnp.concatenate([selected_blocks, tail_blocks[:, None]], axis=1),
            query_positions=query_positions,
            state=updated_state,
        )


@dataclass(frozen=True, kw_only=True)
class SparseAttentionConfig(AttentionConfigBase):
    indexer_config: SparseAttentionIndexerConfig

    def __post_init__(self) -> None:
        if not self.is_causal:
            raise ValueError("Sparse attention must be causal")
        if self.has_sinks:
            raise ValueError("Sparse attention does not support attention sinks")
        if self.sliding_window_size is not None:
            raise ValueError("Sparse attention does not support sliding windows")
        if self.is_kv_sharing:
            raise ValueError("Sparse attention does not support KV sharing")
        if self.normalize_values:
            raise ValueError("Sparse attention does not support value normalization")

    def init(self, initializer: Initializer, model_dim: int) -> "SparseAttention":
        parameters = self.init_parameters(initializer, model_dim)
        return SparseAttention(
            config=self,
            sharding_config=initializer.sharding_config,
            qkvg_projection=parameters.qkvg_projection,
            out_projection=parameters.out_projection,
            query_norm=parameters.query_norm,
            key_norm=parameters.key_norm,
            sinks=parameters.sinks,
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
    scale: float | None,
    logit_soft_cap: float | None,
    forward_pass_config: MixerForwardPassConfig,
) -> Float[Array, "dst_tokens heads head_channels"]:
    num_src_tokens, _, _ = keys.shape
    match forward_pass_config.attention_implementation:
        case AttentionImplementation.PALLAS:
            return pallas_sparse_attention(queries, keys, values, rows, scale, logit_soft_cap)
        case AttentionImplementation.XLA:
            return xla_sparse_attention(queries, keys, values, rows, scale, logit_soft_cap)
        case AttentionImplementation.STABLE_REDUCTION:
            return stable_reduction_attention(
                queries,
                keys,
                values,
                bias=None,
                mask=sparse_rows_mask(rows, num_src_tokens),
                scale=scale,
                logit_soft_cap=logit_soft_cap,
                tile_size=forward_pass_config.attention_tile_size,
                accumulation_dtype=forward_pass_config.attention_accumulation_dtype,
            )


class SparseAttention(AttentionBase[SparseAttentionConfig, SparseAttentionStateLayer]):
    indexer: SparseAttentionIndexer

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

        projection = self.project(inputs, positional_embeddings, forward_pass_config, keychain=qkvg_keychain)
        if isinstance(projection, SharedKeyValueProjection):
            raise NotImplementedError("Sparse attention does not support KV sharing")
        queries, gate, keys, values = projection

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
            scale=self.config.scale,
            logit_soft_cap=self.config.logit_soft_cap,
            forward_pass_config=forward_pass_config,
        )
        result = self.output_projection(attention_output, gate, forward_pass_config, keychain=out_keychain)

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
