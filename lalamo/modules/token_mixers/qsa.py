from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.flow import (
    Context,
    MixerForwardPassConfig,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings, RoPEConfig
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

from .attention import Attention, AttentionConfig
from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, PagedKVCacheLayer, PagedKVCachePool, StaticKVCacheLayer


class QSAState(StateLayerBase):
    cache: KVCacheLayer
    index_keys: Float[Array, "*batch tokens index_channels"]
    positions: Int[Array, "*batch tokens"]


class QSACachePool(StateLayerBase):
    cache: PagedKVCachePool
    index_keys: Float[Array, "pages page_size index_channels"]
    positions: Int[Array, "pages page_size"]

    def write_pages(self, page_indices: Int[Array, "batch pages"], state: QSAState) -> "QSACachePool":
        page_size = self.cache.page_size
        batch, pages = page_indices.shape
        end = pages * page_size
        return QSACachePool(
            self.cache.write_pages(page_indices, state.cache.keys[:, :end], state.cache.values[:, :end]),
            self.index_keys.at[page_indices].set(state.index_keys[:, :end].reshape(batch, pages, page_size, -1)),
            self.positions.at[page_indices].set(state.positions[:, :end].reshape(batch, pages, page_size)),
        )

    def read_pages(
        self,
        page_indices: Int[Array, "batch pages"],
        state: QSAState,
        lengths: Int[Array, " batch"],
    ) -> QSAState:
        assert isinstance(state.cache, StaticKVCacheLayer)
        keys, values = self.cache.read_pages(page_indices)
        batch, pages = page_indices.shape
        end = pages * self.cache.page_size
        return QSAState(
            replace(
                state.cache,
                keys=state.cache.keys.at[:, :end].set(keys),
                values=state.cache.values.at[:, :end].set(values),
                current_length=lengths,
            ),
            state.index_keys.at[:, :end].set(self.index_keys[page_indices].reshape(batch, end, -1)),
            state.positions.at[:, :end].set(self.positions[page_indices].reshape(batch, end)),
        )


class QSAPagedState(StateLayerBase):
    cache: PagedKVCacheLayer
    index_keys: Float[Array, "pages page_size index_channels"]
    positions: Int[Array, "pages page_size"]


@dataclass(frozen=True)
class QSAConfig(TokenMixerConfig):
    attention_config: AttentionConfig
    indexer_projection_config: LinearConfig
    indexer_norm_config: NormalizationConfig
    indexer_num_heads: int
    indexer_head_dim: int
    token_budget: int
    compress_ratio: int
    rope_config: RoPEConfig

    def __post_init__(self) -> None:
        if (
            self.compress_ratio < 1
            or self.token_budget < self.compress_ratio
            or self.token_budget % self.compress_ratio
        ):
            raise ValueError("QSA requires a positive compression ratio and a whole selected-block budget.")
        if self.rope_config.head_dim > self.indexer_head_dim:
            raise ValueError("QSA rotary width cannot exceed the indexer head width.")
        attention = self.attention_config
        if (
            attention.scale not in (None, attention.head_dim**-0.5)
            or attention.logit_soft_cap is not None
            or attention.sliding_window_size is not None
        ):
            raise ValueError("QSA requires standard scaled attention without logit capping or sliding windows.")
        if (
            not self.attention_config.is_causal
            or self.attention_config.has_sinks
            or self.attention_config.kv_source is not None
            or self.attention_config.rope_config is not None
        ):
            raise ValueError("QSA requires causal, state-owning attention without sinks or a separate RoPE.")

    def init(self, initializer: Initializer, model_dim: int) -> "QwenSparseAttention":
        return QwenSparseAttention(
            config=self,
            sharding_config=initializer.sharding_config,
            attention=self.attention_config.init(initializer, model_dim),
            indexer_projection=self.indexer_projection_config.init(
                initializer,
                model_dim,
                (self.indexer_num_heads * self.indexer_head_dim, self.indexer_head_dim),
                has_biases=False,
            ),
            query_norm=self.indexer_norm_config.init(initializer, self.indexer_head_dim),
            key_norm=self.indexer_norm_config.init(initializer, self.indexer_head_dim),
        )


class QwenSparseAttention(TokenMixerBase[QSAConfig, QSAState | QSAPagedState]):
    attention: Attention
    indexer_projection: Linear
    query_norm: Normalization
    key_norm: Normalization

    @property
    def model_dim(self) -> int:
        return self.attention.model_dim

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> QSAState:
        return QSAState(
            self.attention.init_static_state(capacity, dtype),
            jax.device_put(
                jnp.zeros((capacity, self.config.indexer_head_dim), dtype=dtype),
                self.sharding_config.make_sharding((None, None)),
            ),
            jax.device_put(jnp.zeros((capacity,), dtype=jnp.int32), self.sharding_config.make_sharding((None,))),
        )

    def init_pool(self, total_pages: int, page_size: int, dtype: DTypeLike) -> QSACachePool:
        def zeros(shape: tuple[int, ...], dtype: DTypeLike) -> Array:
            return jnp.zeros(shape, dtype=dtype, device=self.sharding_config.make_sharding((None,) * len(shape)))

        shape = (self.attention.config.num_groups, total_pages + 1, page_size, self.attention.config.head_dim)
        return QSACachePool(
            PagedKVCachePool(zeros(shape, dtype), zeros(shape, dtype)),
            zeros((total_pages + 1, page_size, self.config.indexer_head_dim), dtype),
            zeros((total_pages + 1, page_size), jnp.int32),
        )

    def _project(
        self,
        inputs: Float[Array, "tokens channels"],
        positions: Int[Array, " tokens"],
        forward_pass_config: MixerForwardPassConfig,
        keychain: Keychain,
    ) -> tuple[Array, Array, Array, Array | None, Array, Array]:
        attention_keychain, indexer_keychain = keychain.split()
        embeddings = self.config.rope_config.compute_positional_embeddings(positions).astype(
            forward_pass_config.rope_dtype
        )
        queries, keys, values, gate = self.attention.project_heads(
            inputs,
            embeddings,
            forward_pass_config,
            keychain=attention_keychain,
        )
        assert keys is not None and values is not None
        index_queries, index_keys = call_vmapped(
            self.indexer_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=indexer_keychain,
        )
        index_queries = index_queries.reshape(-1, self.config.indexer_num_heads, self.config.indexer_head_dim)
        index_queries = call_vmapped_twice(self.query_norm, index_queries)
        index_queries = call_vmapped(embeddings.apply, index_queries, in_axes=1, out_axes=1)
        return queries, keys, values, gate, index_queries, index_keys

    def _index_blocks(
        self,
        index_keys: Float[Array, "capacity index_channels"],
        positions: Int[Array, " capacity"],
        valid: Array,
        forward_pass_config: MixerForwardPassConfig,
    ) -> tuple[Array, Array]:
        capacity = index_keys.shape[0]
        ratio = self.config.compress_ratio
        logical_indices = jnp.nonzero(valid, size=capacity, fill_value=0)[0]
        ordered = index_keys[logical_indices]
        ordered = jnp.pad(ordered, ((0, (-capacity) % ratio), (0, 0)))
        pooled = ordered.reshape(-1, ratio, self.config.indexer_head_dim).astype(jnp.float32).mean(axis=1)
        pooled = call_vmapped(self.key_norm, pooled.astype(index_keys.dtype))
        block_positions = positions[logical_indices[::ratio]]
        pooled = (
            self.config.rope_config.compute_positional_embeddings(block_positions)
            .astype(forward_pass_config.rope_dtype)
            .apply(pooled)
        )
        return pooled, logical_indices

    def selected_token_indices(
        self,
        query: Float[Array, "heads index_channels"],
        block_keys: Float[Array, "blocks index_channels"],
        logical_indices: Int[Array, " capacity"],
        visible_tokens: Int[Array, ""],
    ) -> Int[Array, " selected_tokens"]:
        ratio = self.config.compress_ratio
        complete_blocks = visible_tokens // ratio
        scores = jnp.einsum(
            "hd,bd->hb",
            query.astype(jnp.float32),
            block_keys.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        scores = jax.nn.relu(scores).sum(axis=0) / self.config.indexer_head_dim**0.5
        scores = jnp.where(jnp.arange(scores.shape[0]) < complete_blocks, scores, -jnp.inf)
        _, selected = jax.lax.top_k(scores, min(self.config.token_budget // ratio, scores.shape[0]))
        block_tokens = (selected[:, None] * ratio + jnp.arange(ratio)).reshape(-1)
        block_valid = jnp.repeat(selected < complete_blocks, ratio)
        tail = complete_blocks * ratio + jnp.arange(ratio - 1)
        logical = jnp.concatenate((block_tokens, tail))
        valid = jnp.concatenate((block_valid, tail < visible_tokens))
        physical = logical_indices[jnp.minimum(logical, logical_indices.shape[0] - 1)]
        return jnp.where(valid, physical, -1)

    def _attend(
        self,
        queries: Array,
        index_queries: Array,
        block_keys: Array,
        logical_indices: Array,
        visible_tokens: Array,
        read_kv: Callable[[Array], tuple[Array, Array]],
        forward_pass_config: MixerForwardPassConfig,
    ) -> Array:
        def attend_one(arguments: tuple[Array, Array, Array]) -> Array:
            query, index_query, length = arguments
            selected = self.selected_token_indices(index_query, block_keys, logical_indices, length)
            keys, values = read_kv(jnp.maximum(selected, 0))
            groups = self.attention.config.num_groups
            query = query.reshape(groups, -1, self.attention.config.head_dim)
            scores = jnp.einsum(
                "ghd,sgd->ghs",
                query,
                keys,
                precision=forward_pass_config.matmul_config.precision,
                preferred_element_type=jnp.float32,
            )
            scores = scores * self.attention.config.head_dim**-0.5
            valid = selected >= 0
            maximum = jnp.max(jnp.where(valid, scores, -jnp.inf), axis=-1, keepdims=True)
            maximum = jnp.where(jnp.any(valid), maximum, 0)
            weights = jnp.where(valid, jnp.exp(jnp.where(valid, scores - maximum, -jnp.inf)), 0)
            weights = weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), jnp.finfo(jnp.float32).tiny)
            outputs = jnp.einsum(
                "ghs,sgd->ghd",
                weights.astype(values.dtype),
                values,
                precision=forward_pass_config.matmul_config.precision,
                preferred_element_type=jnp.float32,
            )
            return outputs.reshape(-1).astype(values.dtype)

        return jax.lax.map(attend_one, (queries, index_queries, visible_tokens), batch_size=256)

    def __call__(
        self,
        inputs: Float[Array, "tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: QSAState | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " tokens"] | None = None,
        *,
        keychain: Keychain,
        token_positions: Int[Array, " tokens"] | None = None,
    ) -> TokenMixerResult[QSAState]:
        if token_positions is None or positional_embeddings is not None:
            raise ValueError("QSA owns RoPE and requires raw token_positions.")
        if attention_parent_indices is not None:
            raise ValueError("QSA requires a causal prefix and owns its KV cache.")
        projection_keychain, output_keychain = keychain.split()
        queries, keys, values, gate, index_queries, index_keys = self._project(
            inputs,
            token_positions,
            forward_pass_config,
            projection_keychain,
        )
        length = inputs.shape[0]
        if length_without_padding is not None:
            length = length_without_padding
        prefix_valid = 0
        if state is None:
            cache = DynamicKVCacheLayer.init(
                has_sinks=False,
                keys=keys.astype(values.dtype),
                values=values,
                length=length,
            )
            all_index_keys, all_positions = index_keys, token_positions
        else:
            prefix = state.cache.current_prefix_length()
            prefix_valid = prefix
            if state.cache.padding_mask is not None:
                prefix_valid = state.cache.padding_mask.sum()
            cache = state.cache.extend(keys.astype(values.dtype), values, added_length=length)
            if isinstance(state.cache, DynamicKVCacheLayer):
                all_index_keys = jnp.concatenate((state.index_keys, index_keys))
                all_positions = jnp.concatenate((state.positions, token_positions))
            else:
                indices = prefix + jnp.arange(inputs.shape[0])
                all_index_keys = state.index_keys.at[indices].set(
                    index_keys.astype(state.index_keys.dtype), mode="drop"
                )
                all_positions = state.positions.at[indices].set(token_positions, mode="drop")
        valid = cache.padding_mask
        if valid is None:
            valid = jnp.ones((cache.keys.shape[0],), dtype=jnp.bool_)
        block_keys, logical_indices = self._index_blocks(all_index_keys, all_positions, valid, forward_pass_config)
        visible = prefix_valid + jnp.minimum(jnp.arange(inputs.shape[0]) + 1, length)
        outputs = self._attend(
            queries,
            index_queries,
            block_keys,
            logical_indices,
            visible,
            lambda selected: (cache.keys[selected], cache.values[selected]),
            forward_pass_config,
        )
        if gate is not None:
            outputs = outputs * jax.nn.sigmoid(gate)
        (outputs,) = call_vmapped(
            self.attention.out_projection,
            outputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=output_keychain,
        )
        updated = None
        if return_updated_state:
            updated = QSAState(cache, all_index_keys, all_positions)
        return TokenMixerResult(outputs, updated)

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *,
        state: QSAState | QSAPagedState | None,
        ctx: Context,
        keychain: Keychain,
    ) -> TokenMixerResult[QSAState | QSAPagedState]:
        forward_pass_config = ctx.forward_pass_config.mixer_forward_pass_config
        if isinstance(state, QSAPagedState):
            return self.paged_decode(x, state, ctx.token_positions, forward_pass_config, keychain=keychain)

        def call_row(
            inputs: Float[Array, "tokens channels"],
            cache: QSAState | None,
            length: Int[Array, ""] | None,
            parents: Int[Array, " tokens"] | None,
            positions: Int[Array, " tokens"],
            *,
            keychain: Keychain,
        ) -> TokenMixerResult[QSAState]:
            return self(
                inputs,
                None,
                cache,
                return_updated_state=ctx.return_updated_state,
                length_without_padding=length,
                forward_pass_config=forward_pass_config,
                attention_parent_indices=parents,
                keychain=keychain,
                token_positions=positions,
            )

        outputs, updated_state = call_vmapped(
            call_row,
            x,
            state,
            ctx.lengths,
            ctx.parents,
            ctx.token_positions,
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        return TokenMixerResult(outputs, updated_state)

    def paged_decode(
        self,
        inputs: Float[Array, "batch 1 channels"],
        state: QSAPagedState,
        token_positions: Int[Array, "batch 1"],
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[QSAPagedState]:
        projection_keychain, output_keychain = keychain.split()
        queries, keys, values, gate, index_queries, index_keys = call_vmapped(
            self._project,
            inputs,
            token_positions,
            forward_pass_config=forward_pass_config,
            keychain=projection_keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        rows = jnp.arange(inputs.shape[0])
        pages = state.cache.block_tables[rows, state.cache.lengths // state.cache.page_size]
        offsets = state.cache.lengths % state.cache.page_size
        all_index_keys = state.index_keys.at[pages, offsets].set(index_keys[:, 0].astype(state.index_keys.dtype))
        positions = state.positions.at[pages, offsets].set(token_positions[:, 0])
        cache = state.cache.append(keys[:, 0].astype(values.dtype), values[:, 0])

        def attend_row(arguments: tuple[Array, Array, Array]) -> Array:
            row, query, index_query = arguments
            table = cache.block_tables[row]
            raw_keys = all_index_keys[table].reshape(-1, self.config.indexer_head_dim)
            raw_positions = positions[table].reshape(-1)
            valid = jnp.arange(raw_keys.shape[0]) < cache.lengths[row]
            pooled, logical = self._index_blocks(raw_keys, raw_positions, valid, forward_pass_config)

            def read_kv(selected: Array) -> tuple[Array, Array]:
                physical_pages = table[selected // cache.page_size]
                physical_offsets = selected % cache.page_size
                selected_keys, selected_values = (
                    jnp.swapaxes(array[:, physical_pages, physical_offsets], 0, 1)
                    for array in (cache.keys, cache.values)
                )
                return selected_keys, selected_values

            return self._attend(
                query, index_query, pooled, logical, cache.lengths[row][None], read_kv, forward_pass_config
            )

        outputs = jax.lax.map(attend_row, (rows, queries, index_queries))
        if gate is not None:
            outputs = outputs * jax.nn.sigmoid(gate)
        (outputs,) = call_vmapped_twice(
            self.attention.out_projection,
            outputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=output_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        return TokenMixerResult(outputs, QSAPagedState(cache, all_index_keys, positions))
