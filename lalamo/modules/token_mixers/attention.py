from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.kernels.attention import (
    paged_decode_attention,
    pallas_decode_attention,
    pallas_sparse_attention,
    sparse_rows_mask,
    stable_reduction_attention,
    xla_attention,
    xla_sparse_attention,
)
from lalamo.module import Keychain, LalamoConfig, LogicalAxis
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

from .convolutions import CausalConv, CausalConvConfig
from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, PagedKVCacheLayer, StaticKVCacheLayer

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
    "ConvAttentionState",
    "L2ClipConfig",
    "LogScalingConfig",
    "RelativePositionBiasConfig",
    "SparseMask",
]


@dataclass(frozen=True)
class L2ClipConfig(LalamoConfig):
    """Scales each head vector down to `threshold` when its L2 norm is larger, after an optional learned scale."""

    threshold: float
    has_scale: bool

    def init_scale(
        self, initializer: Initializer, num_heads: int, head_dim: int
    ) -> Float[Array, "heads channels"] | None:
        if not self.has_scale:
            return None
        return initializer.ones((num_heads, head_dim), dtype=jnp.float32)

    def __call__(
        self,
        heads: Float[Array, "tokens heads channels"],
        scale: Float[Array, "heads channels"] | None,
    ) -> Float[Array, "tokens heads channels"]:
        if (scale is None) == self.has_scale:
            raise ValueError("The L2 clip scale must be present exactly when has_scale is set.")
        upcasted = heads.astype(jnp.float32)
        if scale is not None:
            upcasted = upcasted * scale.astype(jnp.float32)
        norm = jnp.sqrt(jnp.sum(jnp.square(upcasted), axis=-1, keepdims=True))
        return (upcasted * jnp.minimum(self.threshold / (norm + 1e-6), 1.0)).astype(heads.dtype)


@dataclass(frozen=True)
class LogScalingConfig(LalamoConfig):
    """Position-dependent scale `1 + alpha * log(max((position + 1) / n_floor, 1))`."""

    n_floor: int
    alpha: float

    def __call__(self, positions: Int[Array, " tokens"]) -> Float[Array, " tokens"]:
        ratio = (positions.astype(jnp.float32) + 1) / self.n_floor
        return 1.0 + self.alpha * jnp.log(jnp.maximum(ratio, 1.0))


@dataclass(frozen=True)
class RelativePositionBiasConfig(LalamoConfig):
    """Per-token relative states mix learned bias-versus-distance profiles into one logit bias per distance.

    The bias is zero outside `0 <= distance < extent`.
    """

    dim: int
    extent: int
    clip_config: L2ClipConfig


class ConvAttentionState(StateLayerBase):
    kv_cache: KVCacheLayer
    key_conv_state: Float[Array, "*batch kernel_minus_one channels"] | None
    value_conv_state: Float[Array, "*batch kernel_minus_one channels"] | None


def _relative_position_bias(
    relative_logits: Float[Array, "dst_tokens heads extent"],
    query_positions: Int[Array, " dst_tokens"],
    key_positions: Int[Array, " src_tokens"],
) -> Float[Array, "heads dst_tokens src_tokens"]:
    _, _, extent = relative_logits.shape
    distances = query_positions[:, None] - key_positions[None, :]
    is_biased = (distances >= 0) & (distances < extent) & (key_positions[None, :] >= 0)
    gathered = jnp.take_along_axis(
        rearrange(relative_logits, "dst heads extent -> heads dst extent"),
        jnp.clip(distances, 0, extent - 1)[None],
        axis=-1,
    )
    return jnp.where(is_biased[None], gathered, 0.0)


def _rms_normalize(
    inputs: Float[Array, "... channels"],
    eps: float,
) -> Float[Array, "... channels"]:
    upcasted_inputs = inputs.astype(jnp.float32)
    variance = jnp.mean(jnp.square(upcasted_inputs), axis=-1, keepdims=True)
    return (upcasted_inputs * jax.lax.rsqrt(variance + eps)).astype(inputs.dtype)


def _attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    *,
    bias: Float[Array, "heads dst_tokens src_tokens"] | None,
    mask: Bool[Array, "dst_tokens src_tokens"],
    scale: float | None,
    logit_soft_cap: float | None,
    forward_pass_config: MixerForwardPassConfig,
) -> Float[Array, "dst_tokens heads head_channels"]:
    match forward_pass_config.attention_implementation:
        case AttentionImplementation.PALLAS:
            return pallas_decode_attention(
                queries,
                keys,
                values,
                bias,
                mask,
                scale,
                logit_soft_cap,
            )
        case AttentionImplementation.XLA:
            return xla_attention(
                queries,
                keys,
                values,
                bias,
                mask,
                scale,
                logit_soft_cap,
            )
        case AttentionImplementation.STABLE_REDUCTION:
            return stable_reduction_attention(
                queries,
                keys,
                values,
                bias=bias,
                mask=mask,
                scale=scale,
                logit_soft_cap=logit_soft_cap,
                tile_size=forward_pass_config.attention_tile_size,
                accumulation_dtype=forward_pass_config.attention_accumulation_dtype,
            )


class SparseMask(NamedTuple):
    positions: Int[Array, "dst_tokens selected_tokens"]

    def rows(self, kv_cache: KVCacheLayer) -> Int[Array, "dst_tokens selected_tokens"]:
        num_slots, _, _ = kv_cache.keys.shape
        return jnp.where(self.positions >= 0, kv_cache.slots_for_positions(self.positions), num_slots)


def sparse_attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    *,
    rows: Int[Array, "dst_tokens selected_tokens"],
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


AttentionResult = TokenMixerResult[KVCacheLayer]


@dataclass(frozen=True)
class AttentionConfig(TokenMixerConfig):
    # Unified QKVG projection; ungated attention omits the G output segment.
    qkvg_projection_config: LinearConfig
    out_projection_config: LinearConfig

    query_norm_config: NormalizationConfig | None
    key_norm_config: NormalizationConfig | None

    num_heads: int
    num_groups: int
    head_dim: int
    is_causal: bool
    scale: float | None
    sliding_window_size: int | None

    logit_soft_cap: float | None
    has_sinks: bool
    has_qkvg_biases: bool
    has_out_biases: bool
    # Query-width sigmoid gate appended as the final fused projection segment.
    has_gate: bool = False
    # Scale-free RMS normalization on values
    normalize_values: bool = False
    is_kv_sharing: bool = False
    # Queries, keys and values project separate inputs: the query projection also yields the relative states, the
    # gate and the token sinks; the key and value projections also yield their dynamic convolution taps.
    separate_inputs: bool = False
    query_clip_config: L2ClipConfig | None = None
    key_clip_config: L2ClipConfig | None = None
    log_scaling_config: LogScalingConfig | None = None
    relative_position_bias_config: RelativePositionBiasConfig | None = None
    # A sink logit per token and head, projected from the query input, with a zero value.
    has_token_sinks: bool = False
    key_conv_config: CausalConvConfig | None = None
    value_conv_config: CausalConvConfig | None = None

    def __post_init__(self) -> None:
        separate_input_features = (
            self.query_clip_config,
            self.key_clip_config,
            self.log_scaling_config,
            self.relative_position_bias_config,
            self.key_conv_config,
            self.value_conv_config,
        )
        if not self.separate_inputs and (
            self.has_token_sinks or any(feature is not None for feature in separate_input_features)
        ):
            raise ValueError(
                "Clipping, log scaling, relative bias, token sinks and convolutions need separate inputs."
            )
        if self.separate_inputs and self.is_kv_sharing:
            raise ValueError("Attention with separate inputs cannot share another layer's cache.")
        if self.has_token_sinks and self.has_sinks:
            raise ValueError("Attention has either learned per-head sinks or projected token sinks, not both.")

    @property
    def has_sink_slot(self) -> bool:
        return self.has_sinks or self.has_token_sinks

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "Attention":
        if self.separate_inputs:
            return _init_separate_input_attention(self, initializer, model_dim)
        q_output_dim = self.num_heads * self.head_dim
        if self.is_kv_sharing:
            output_dims = (q_output_dim,)
        else:
            output_dims = (
                q_output_dim,
                self.num_groups * self.head_dim,
                self.num_groups * self.head_dim,
            )
        if self.has_gate:
            output_dims = (*output_dims, q_output_dim)

        return _init_attention(self, initializer, model_dim, output_dims)


def _init_attention(
    config: AttentionConfig,
    initializer: Initializer,
    model_dim: int,
    output_dims: tuple[int, ...],
) -> "Attention":
    qkvg_projection = config.qkvg_projection_config.init(
        initializer,
        input_dim=model_dim,
        output_dims=output_dims,
        has_biases=config.has_qkvg_biases,
    )
    out_projection = config.out_projection_config.init(
        initializer,
        config.num_heads * config.head_dim,
        (model_dim,),
        has_biases=config.has_out_biases,
    )

    if config.query_norm_config is not None:
        query_norm = config.query_norm_config.init(
            initializer,
            input_dim=config.head_dim,
        )
    else:
        query_norm = None

    if config.key_norm_config is not None and not config.is_kv_sharing:
        key_norm = config.key_norm_config.init(
            initializer,
            input_dim=config.head_dim,
        )
    else:
        key_norm = None

    if config.has_sinks:
        sinks = initializer.zeros((config.num_heads,))
    else:
        sinks = None

    return Attention(
        config=config,
        sharding_config=initializer.sharding_config,
        qkvg_projection=qkvg_projection,
        out_projection=out_projection,
        query_norm=query_norm,
        key_norm=key_norm,
        sinks=sinks,
    )


def _conv_output_dims(conv_config: CausalConvConfig | None) -> tuple[int, ...]:
    if conv_config is None or conv_config.num_dynamic_groups is None:
        return ()
    return (conv_config.num_dynamic_groups * conv_config.kernel_size,)


def _init_separate_input_attention(
    config: AttentionConfig,
    initializer: Initializer,
    model_dim: int,
) -> "Attention":
    query_dim = config.num_heads * config.head_dim
    key_value_dim = config.num_groups * config.head_dim
    relative_config = config.relative_position_bias_config

    query_output_dims = (query_dim,)
    if relative_config is not None:
        query_output_dims = (*query_output_dims, config.num_heads * relative_config.dim)
    if config.has_gate:
        query_output_dims = (*query_output_dims, query_dim)
    if config.has_token_sinks:
        query_output_dims = (*query_output_dims, config.num_heads)

    def init_projection(output_dims: tuple[int, ...]) -> Linear:
        return config.qkvg_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=output_dims,
            has_biases=config.has_qkvg_biases,
        )

    def init_conv(conv_config: CausalConvConfig | None) -> CausalConv | None:
        if conv_config is None:
            return None
        return conv_config.init(initializer, key_value_dim)

    return Attention(
        config=config,
        sharding_config=initializer.sharding_config,
        qkvg_projection=None,
        out_projection=config.out_projection_config.init(
            initializer,
            query_dim,
            (model_dim,),
            has_biases=config.has_out_biases,
        ),
        query_norm=config.query_norm_config.init(initializer, config.head_dim) if config.query_norm_config else None,
        key_norm=config.key_norm_config.init(initializer, config.head_dim) if config.key_norm_config else None,
        sinks=initializer.zeros((config.num_heads,)) if config.has_sinks else None,
        query_projection=init_projection(query_output_dims),
        key_projection=init_projection((key_value_dim, *_conv_output_dims(config.key_conv_config))),
        value_projection=init_projection((key_value_dim, *_conv_output_dims(config.value_conv_config))),
        query_clip_scale=(
            config.query_clip_config.init_scale(initializer, config.num_heads, config.head_dim)
            if config.query_clip_config is not None
            else None
        ),
        key_clip_scale=(
            config.key_clip_config.init_scale(initializer, config.num_groups, config.head_dim)
            if config.key_clip_config is not None
            else None
        ),
        relative_position_profiles=(
            initializer.normal(1.0, (relative_config.dim, relative_config.extent), dtype=jnp.float32)
            if relative_config is not None
            else None
        ),
        relative_clip_scale=(
            relative_config.clip_config.init_scale(initializer, config.num_heads, relative_config.dim)
            if relative_config is not None
            else None
        ),
        key_conv=init_conv(config.key_conv_config),
        value_conv=init_conv(config.value_conv_config),
    )


class Attention(TokenMixerBase[AttentionConfig, KVCacheLayer | ConvAttentionState]):
    qkvg_projection: Linear | None
    out_projection: Linear

    query_norm: Normalization | None
    key_norm: Normalization | None

    sinks: Float[Array, " heads"] | None

    query_projection: Linear | None = None
    key_projection: Linear | None = None
    value_projection: Linear | None = None
    query_clip_scale: Float[Array, "heads head_channels"] | None = None
    key_clip_scale: Float[Array, "groups head_channels"] | None = None
    relative_position_profiles: Float[Array, "relative_channels extent"] | None = None
    relative_clip_scale: Float[Array, "heads relative_channels"] | None = None
    key_conv: CausalConv | None = None
    value_conv: CausalConv | None = None

    @property
    def model_dim(self) -> int:
        if self.qkvg_projection is None:
            assert self.query_projection is not None
            return self.query_projection.input_dim
        return self.qkvg_projection.input_dim

    @property
    def use_sliding_window(self) -> bool:
        return self.config.sliding_window_size is not None

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        if self.use_sliding_window:
            return PositionalEmbeddingSelector.LOCAL
        return PositionalEmbeddingSelector.GLOBAL

    @property
    def has_sinks(self) -> bool:
        return self.sinks is not None

    @eqx.filter_jit
    def _prepare_heads(
        self,
        projection: Float[Array, "tokens channels"],
        num_heads: int,
        norm: Normalization | None,
        positional_embeddings: PositionalEmbeddings | None,
    ) -> Float[Array, "tokens heads head_channels"]:
        heads = rearrange(
            projection,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=num_heads,
            head_channels=self.config.head_dim,
        )
        if norm is not None:
            heads = call_vmapped_twice(norm, heads)
        if positional_embeddings is not None:
            heads = call_vmapped(positional_embeddings.apply, heads, in_axes=1, out_axes=1)
        return heads

    def project_heads(
        self,
        inputs: Float[Array, "tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "tokens heads head_channels"],
        Float[Array, "tokens groups head_channels"] | None,
        Float[Array, "tokens groups head_channels"] | None,
        Float[Array, "tokens indexer_channels"] | None,
        Float[Array, "tokens heads*head_channels"] | None,
    ]:
        if self.qkvg_projection is None:
            raise ValueError("Attention with separate inputs projects them in `_call_separate_inputs`.")
        projections = call_vmapped(
            self.qkvg_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        queries = self._prepare_heads(projections[0], self.config.num_heads, self.query_norm, positional_embeddings)
        gate = projections[-1] if self.config.has_gate else None
        if self.config.is_kv_sharing:
            return queries, None, None, None, gate
        qkvi_projections = projections[:-1] if self.config.has_gate else projections
        _, keys, values, *indexer = qkvi_projections
        indexer_projection = None
        if indexer:
            (indexer_projection,) = indexer
        keys = self._prepare_heads(keys, self.config.num_groups, self.key_norm, positional_embeddings)
        values = rearrange(
            values,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_groups,
            head_channels=self.config.head_dim,
        )
        if self.config.normalize_values:
            values = _rms_normalize(values, eps=1e-6)
        return queries, keys, values, indexer_projection, gate

    def project_key_value_heads(
        self,
        inputs: Float[Array, "new_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "new_tokens groups head_channels"],
        Float[Array, "new_tokens groups head_channels"],
    ]:
        if self.config.is_kv_sharing:
            raise ValueError("KV-sharing attention layers do not own key/value projections.")
        _, keys, values, _, _ = self.project_heads(
            inputs, positional_embeddings, forward_pass_config, keychain=keychain
        )
        assert keys is not None and values is not None
        return keys, values

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"] | Float[Array, "suffix_tokens roles channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: KVCacheLayer | ConvAttentionState | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        reuse_cache: bool = False,
        *,
        keychain: Keychain,
        sparse_mask: SparseMask | None = None,
    ) -> TokenMixerResult[KVCacheLayer | ConvAttentionState]:
        if self.config.separate_inputs:
            if sparse_mask is not None or attention_parent_indices is not None:
                raise NotImplementedError("Attention with separate inputs supports neither sparse nor tree masks.")
            if positional_embeddings is not None:
                raise NotImplementedError("Attention with separate inputs does not apply positional embeddings.")
            return self._call_separate_inputs(
                inputs,
                state,
                return_updated_state,
                length_without_padding,
                forward_pass_config,
                keychain=keychain,
            )
        if isinstance(state, ConvAttentionState):
            raise TypeError("Only attention with separate inputs has convolution states.")
        qkvg_keychain, out_keychain = keychain.split(2)
        queries, keys, values, _, gate = self.project_heads(
            inputs, positional_embeddings, forward_pass_config, keychain=qkvg_keychain
        )
        return self.attend(
            queries,
            keys,
            values,
            gate,
            state,
            return_updated_state,
            length_without_padding,
            forward_pass_config,
            attention_parent_indices,
            reuse_cache,
            keychain=out_keychain,
            sparse_mask=sparse_mask,
        )

    def attend(
        self,
        queries: Float[Array, "tokens heads head_channels"],
        keys: Float[Array, "tokens groups head_channels"] | None,
        values: Float[Array, "tokens groups head_channels"] | None,
        gate: Float[Array, "tokens heads*head_channels"] | None,
        state: KVCacheLayer | None,
        return_updated_state: bool,
        length_without_padding: Int[Array, ""] | int | None,
        forward_pass_config: MixerForwardPassConfig,
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        reuse_cache: bool = False,
        *,
        keychain: Keychain,
        sparse_mask: SparseMask | None = None,
    ) -> AttentionResult:
        if sparse_mask is not None and attention_parent_indices is not None:
            raise ValueError("A sparse mask cannot be combined with tree attention.")
        if sparse_mask is not None and self.sinks is not None:
            raise ValueError("A sparse mask cannot be combined with attention sinks.")
        assert reuse_cache == self.config.is_kv_sharing, "reuse_cache must match AttentionConfig.is_kv_sharing"
        num_suffix_tokens, _, _ = queries.shape
        if reuse_cache:
            if state is None:
                raise ValueError("a KV-sharing layer must receive the source layer's cache as `state`")
            prefix_length = state.current_prefix_length() - num_suffix_tokens
            updated_state = state
        else:
            assert keys is not None and values is not None
            prefix_length = 0 if state is None else state.current_prefix_length()
            if state is None:
                updated_state = DynamicKVCacheLayer.init(
                    self.has_sinks, keys.astype(values.dtype), values, length=length_without_padding
                )
            else:
                updated_state = state.extend(keys, values, added_length=length_without_padding)

        queries = queries.astype(updated_state.keys.dtype)
        if sparse_mask is not None:
            attention_output = sparse_attention_kernel(
                queries,
                updated_state.keys,
                updated_state.values,
                rows=sparse_mask.rows(updated_state),
                scale=self.config.scale,
                logit_soft_cap=self.config.logit_soft_cap,
                forward_pass_config=forward_pass_config,
            )
        else:
            if attention_parent_indices is not None:
                mask = updated_state.tree_attention_mask(prefix_length, attention_parent_indices)
            else:
                mask = updated_state.attention_mask(
                    num_suffix_tokens,
                    self.config.is_causal,
                    length_without_padding,
                    self.config.sliding_window_size,
                )
            if self.sinks is not None:
                sink_bias = jnp.zeros((self.config.num_heads, *mask.shape), dtype=queries.dtype)
                sink_bias = sink_bias.at[:, :, 0].set(self.sinks[:, None].astype(sink_bias.dtype))
            else:
                sink_bias = None

            attention_output = _attention_kernel(
                queries,
                updated_state.keys,
                updated_state.values,
                bias=sink_bias,
                mask=mask,
                scale=self.config.scale,
                logit_soft_cap=self.config.logit_soft_cap,
                forward_pass_config=forward_pass_config,
            )
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        if gate is not None:
            attention_output = attention_output * jax.nn.sigmoid(gate)
        (result,) = call_vmapped(
            self.out_projection,
            attention_output,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )

        if not return_updated_state:
            updated_state = None
        return AttentionResult(outputs=result, state=updated_state)

    def paged_decode(
        self,
        inputs: Float[Array, "batch 1 channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: PagedKVCacheLayer,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[PagedKVCacheLayer]:
        qkvg_keychain, out_keychain = keychain.split(2)
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        queries, keys, values, _, gate = call_vmapped(
            self.project_heads,
            inputs,
            positional_embeddings,
            forward_pass_config=forward_pass_config,
            keychain=qkvg_keychain,
            added_sharding_axis=batch_axis,
        )
        if keys is not None and values is not None:
            state = state.append(keys[:, 0], values[:, 0])
        queries = queries[:, 0].astype(state.keys.dtype)
        scale = self.config.scale if self.config.scale is not None else self.config.head_dim**-0.5

        attention_output = paged_decode_attention(
            queries,
            state.keys,
            state.values,
            state.block_tables,
            state.lengths,
            scale=scale,
            logit_soft_cap=self.config.logit_soft_cap,
            sinks=self.sinks,
            sliding_window_size=self.config.sliding_window_size,
        )

        attention_output = rearrange(attention_output, "batch heads channels -> batch 1 (heads channels)")
        if gate is not None:
            attention_output *= jax.nn.sigmoid(gate)
        (outputs,) = call_vmapped_twice(
            self.out_projection,
            attention_output,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain,
            added_sharding_axes=(batch_axis, None),
        )
        return TokenMixerResult(outputs, state)

    def _call_separate_inputs(
        self,
        inputs: Float[Array, "suffix_tokens roles channels"],
        state: "ConvAttentionState | KVCacheLayer | None",
        return_updated_state: bool,
        length_without_padding: Int[Array, ""] | int | None,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[KVCacheLayer | ConvAttentionState]:
        assert self.query_projection is not None
        assert self.key_projection is not None
        assert self.value_projection is not None
        num_suffix_tokens, num_roles, _ = inputs.shape
        if num_roles != 3:
            raise ValueError(f"Attention with separate inputs expects query, key and value inputs, got {num_roles}.")
        query_keychain, key_keychain, value_keychain, out_keychain = keychain.split(4)
        matmul_config = forward_pass_config.matmul_config

        query_outputs = list(
            call_vmapped(
                self.query_projection, inputs[:, 0], forward_pass_config=matmul_config, keychain=query_keychain
            )
        )
        key_outputs = list(
            call_vmapped(self.key_projection, inputs[:, 1], forward_pass_config=matmul_config, keychain=key_keychain)
        )
        value_outputs = list(
            call_vmapped(
                self.value_projection, inputs[:, 2], forward_pass_config=matmul_config, keychain=value_keychain
            )
        )
        queries = query_outputs.pop(0)
        relative_states = query_outputs.pop(0) if self.config.relative_position_bias_config is not None else None
        gate = query_outputs.pop(0) if self.config.has_gate else None
        token_sinks = query_outputs.pop(0) if self.config.has_token_sinks else None
        assert not query_outputs

        if isinstance(state, ConvAttentionState):
            kv_cache, key_conv_state, value_conv_state = state.kv_cache, state.key_conv_state, state.value_conv_state
        else:
            kv_cache, key_conv_state, value_conv_state = state, None, None

        def convolve(
            conv: CausalConv | None,
            projections: list[Array],
            conv_state: Float[Array, "kernel_minus_one channels"] | None,
        ) -> tuple[Float[Array, "tokens channels"], Float[Array, "kernel_minus_one channels"] | None]:
            outputs, *taps = projections
            if conv is None:
                assert not taps
                return outputs, None
            coefficient_deltas = (
                rearrange(taps[0], "tokens (groups kernel) -> tokens kernel groups", kernel=conv.kernel_size)
                if taps
                else None
            )
            result = conv(
                outputs,
                conv_state,
                return_updated_state=return_updated_state,
                length_without_padding=length_without_padding,
                coefficient_deltas=coefficient_deltas,
            )
            return result.outputs, result.state

        keys, updated_key_conv_state = convolve(self.key_conv, key_outputs, key_conv_state)
        values, updated_value_conv_state = convolve(self.value_conv, value_outputs, value_conv_state)

        query_heads = rearrange(
            queries, "tokens (heads channels) -> tokens heads channels", heads=self.config.num_heads
        )
        key_heads = rearrange(
            keys, "tokens (groups channels) -> tokens groups channels", groups=self.config.num_groups
        )
        value_heads = rearrange(
            values, "tokens (groups channels) -> tokens groups channels", groups=self.config.num_groups
        )
        if self.query_norm is not None:
            query_heads = call_vmapped_twice(self.query_norm, query_heads)
        if self.key_norm is not None:
            key_heads = call_vmapped_twice(self.key_norm, key_heads)
        if self.config.query_clip_config is not None:
            query_heads = self.config.query_clip_config(query_heads, self.query_clip_scale)
        if self.config.key_clip_config is not None:
            key_heads = self.config.key_clip_config(key_heads, self.key_clip_scale)
        if self.config.normalize_values:
            value_heads = _rms_normalize(value_heads, eps=1e-6)

        if kv_cache is None:
            updated_kv_cache = DynamicKVCacheLayer.init(
                self.config.has_sink_slot,
                key_heads.astype(value_heads.dtype),
                value_heads,
                length=length_without_padding,
            )
        else:
            updated_kv_cache = kv_cache.extend(key_heads, value_heads, added_length=length_without_padding)
        query_positions, key_positions = updated_kv_cache.token_positions(num_suffix_tokens, length_without_padding)

        if self.config.log_scaling_config is not None:
            log_scales = self.config.log_scaling_config(query_positions)[:, None, None]
            query_heads = (query_heads.astype(jnp.float32) * log_scales).astype(query_heads.dtype)
        else:
            log_scales = None

        mask = updated_kv_cache.attention_mask(
            num_suffix_tokens,
            self.config.is_causal,
            length_without_padding,
            self.config.sliding_window_size,
        )
        num_slots = mask.shape[-1]
        bias = jnp.zeros((self.config.num_heads, num_suffix_tokens, num_slots), dtype=jnp.float32)
        relative_config = self.config.relative_position_bias_config
        if relative_config is not None:
            assert relative_states is not None and self.relative_position_profiles is not None
            relative_heads = rearrange(
                relative_states, "tokens (heads channels) -> tokens heads channels", heads=self.config.num_heads
            )
            relative_heads = relative_config.clip_config(relative_heads, self.relative_clip_scale).astype(jnp.float32)
            if log_scales is not None:
                relative_heads = relative_heads * log_scales
            relative_logits = jnp.einsum(
                "thc,ce->the",
                relative_heads,
                self.relative_position_profiles.astype(jnp.float32),
            )
            bias = bias + _relative_position_bias(relative_logits, query_positions, key_positions)
        if self.sinks is not None:
            bias = bias.at[:, :, 0].set(self.sinks[:, None].astype(bias.dtype))
        if token_sinks is not None:
            bias = bias.at[:, :, 0].set(rearrange(token_sinks, "tokens heads -> heads tokens").astype(bias.dtype))

        attention_output = _attention_kernel(
            query_heads.astype(updated_kv_cache.keys.dtype),
            updated_kv_cache.keys,
            updated_kv_cache.values,
            bias=bias,
            mask=mask,
            scale=self.config.scale,
            logit_soft_cap=self.config.logit_soft_cap,
            forward_pass_config=forward_pass_config,
        )
        attention_output = rearrange(attention_output, "tokens heads channels -> tokens (heads channels)")
        if gate is not None:
            attention_output = (
                attention_output.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
            ).astype(attention_output.dtype)
        (result,) = call_vmapped(
            self.out_projection,
            attention_output,
            forward_pass_config=matmul_config,
            keychain=out_keychain,
        )

        if not return_updated_state:
            return TokenMixerResult(outputs=result, state=None)
        if self.key_conv is None and self.value_conv is None:
            return TokenMixerResult(outputs=result, state=updated_kv_cache)
        return TokenMixerResult(
            outputs=result,
            state=ConvAttentionState(
                kv_cache=updated_kv_cache,
                key_conv_state=updated_key_conv_state,
                value_conv_state=updated_value_conv_state,
            ),
        )

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> "StaticKVCacheLayer | ConvAttentionState":
        kv_cache = StaticKVCacheLayer.init(
            self.config.has_sink_slot,
            capacity,
            self.config.num_groups,
            self.config.head_dim,
            dtype,
            self.sharding_config,
        )
        if self.key_conv is None and self.value_conv is None:
            return kv_cache
        return ConvAttentionState(
            kv_cache=kv_cache,
            key_conv_state=self.key_conv.init_static_state() if self.key_conv is not None else None,
            value_conv_state=self.value_conv.init_static_state() if self.value_conv is not None else None,
        )
