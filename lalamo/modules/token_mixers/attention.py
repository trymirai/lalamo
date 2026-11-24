from dataclasses import dataclass, replace
from enum import StrEnum
from typing import NamedTuple

import equinox as eqx
import jax
from einops import einsum, rearrange
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
    QKVInputs,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

from .convolutions import CausalConvResult, SeparableCausalConv, SeparableCausalConvConfig
from .kv_cache import DynamicKVCacheLayer, KeyValueConvState, KVCacheLayer, PagedKVCacheLayer, StaticKVCacheLayer

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
    "KeyValueConvConfig",
    "LogScalingConfig",
    "ProjectionLayout",
    "RelativeBiasConfig",
    "SparseMask",
    "StackedProjectionConfig",
    "aligned_rows",
]


def _rms_normalize(
    inputs: Float[Array, "... channels"],
    eps: float,
) -> Float[Array, "... channels"]:
    upcasted_inputs = inputs.astype(jnp.float32)
    variance = jnp.mean(jnp.square(upcasted_inputs), axis=-1, keepdims=True)
    return (upcasted_inputs * jax.lax.rsqrt(variance + eps)).astype(inputs.dtype)


def aligned_rows(rows: int, alignment: int) -> int:
    return -(-rows // alignment) * alignment


def clip_norm(heads: Float[Array, "*dims channels"], threshold: float) -> Float[Array, "*dims channels"]:
    upcast_heads = heads.astype(jnp.float32)
    norms = jnp.linalg.norm(upcast_heads, axis=-1, keepdims=True)
    return upcast_heads * jnp.minimum(threshold / (norms + 1e-6), 1.0)


@dataclass(frozen=True)
class RelativeBiasConfig(LalamoConfig):
    num_channels: int
    extent: int
    clip_threshold: float


@dataclass(frozen=True)
class LogScalingConfig(LalamoConfig):
    alpha: float
    position_floor: int

    def scales(self, positions: Int[Array, " tokens"]) -> Float[Array, " tokens"]:
        relative_positions = (positions + 1).astype(jnp.float32) / self.position_floor
        return 1 + self.alpha * jnp.log(jnp.maximum(relative_positions, 1.0))


class ProjectionLayout(StrEnum):
    STACKED_QUERY_FIRST = "stacked_query_first"
    GROUPED_WITH_CONV_TAPS = "grouped_with_conv_taps"


@dataclass(frozen=True)
class KeyValueConvConfig(LalamoConfig):
    conv_config: SeparableCausalConvConfig
    kernel_size: int

    def init(self, initializer: Initializer, channels: int) -> SeparableCausalConv:
        return self.conv_config.init(initializer, channels, self.kernel_size)


@dataclass(frozen=True)
class StackedProjectionConfig(LalamoConfig):
    layout: ProjectionLayout
    row_alignment: int
    relative_bias_config: RelativeBiasConfig
    key_value_conv_config: KeyValueConvConfig

    def logical_output_dims(self, num_heads: int, num_groups: int, head_dim: int) -> tuple[tuple[int, ...], ...]:
        query_dim = num_heads * head_dim
        key_value_dim = num_groups * head_dim
        query_dims = (query_dim, query_dim, num_heads * self.relative_bias_config.num_channels, num_heads)
        match self.layout:
            case ProjectionLayout.STACKED_QUERY_FIRST:
                return ((*query_dims, key_value_dim, key_value_dim),)
            case ProjectionLayout.GROUPED_WITH_CONV_TAPS:
                key_value_dims = (key_value_dim, num_groups * self.key_value_conv_config.kernel_size)
                return (query_dims, key_value_dims, key_value_dims)

    def output_dims(self, num_heads: int, num_groups: int, head_dim: int) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(aligned_rows(rows, self.row_alignment) for rows in dims)
            for dims in self.logical_output_dims(num_heads, num_groups, head_dim)
        )


class KeyValueProjection(NamedTuple):
    states: Float[Array, "suffix_tokens channels"]
    taps: Float[Array, "suffix_tokens taps"] | None


class ProjectedHeads(NamedTuple):
    queries: Float[Array, "suffix_tokens heads head_channels"]
    keys: Float[Array, "suffix_tokens groups head_channels"] | None
    values: Float[Array, "suffix_tokens groups head_channels"] | None
    gate: Float[Array, "suffix_tokens query_channels"] | None
    relative_states: Float[Array, "suffix_tokens relative_channels"] | None
    sink_logits: Float[Array, "suffix_tokens heads"] | None
    conv_states: KeyValueConvState | None


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
    query_clip_threshold: float | None = None
    key_clip_threshold: float | None = None
    log_scaling_config: LogScalingConfig | None = None
    stacked_projection_config: StackedProjectionConfig | None = None

    def __post_init__(self) -> None:
        if self.stacked_projection_config is not None and (
            not self.has_sinks or not self.has_gate or self.is_kv_sharing
        ):
            raise ValueError(
                "Stacked projections always project per-token sinks and a gate, and their own keys and values."
            )

    @property
    def supports_paged_decode(self) -> bool:
        return self.stacked_projection_config is None and self.log_scaling_config is None

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "Attention":
        if self.stacked_projection_config is not None:
            qkvg_dims, *key_value_dims = self.stacked_projection_config.output_dims(
                self.num_heads,
                self.num_groups,
                self.head_dim,
            )
            return _init_attention(self, initializer, model_dim, qkvg_dims, tuple(key_value_dims))
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
    key_value_output_dims: tuple[tuple[int, ...], ...] = (),
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

    if config.has_sinks and config.stacked_projection_config is None:
        sinks = initializer.zeros((config.num_heads,))
    else:
        sinks = None

    stacked_config = config.stacked_projection_config
    key_value_dim = config.num_groups * config.head_dim
    return Attention(
        config=config,
        sharding_config=initializer.sharding_config,
        qkvg_projection=qkvg_projection,
        out_projection=out_projection,
        query_norm=query_norm,
        key_norm=key_norm,
        sinks=sinks,
        relative_profiles=(
            initializer.zeros(
                (stacked_config.relative_bias_config.num_channels, stacked_config.relative_bias_config.extent),
                dtype=jnp.float32,
            )
            if stacked_config is not None
            else None
        ),
        key_value_projections=tuple(
            config.qkvg_projection_config.init(
                initializer,
                input_dim=model_dim,
                output_dims=dims,
                has_biases=config.has_qkvg_biases,
            )
            for dims in key_value_output_dims
        ),
        key_conv=(
            stacked_config.key_value_conv_config.init(initializer, key_value_dim)
            if stacked_config is not None
            else None
        ),
        value_conv=(
            stacked_config.key_value_conv_config.init(initializer, key_value_dim)
            if stacked_config is not None
            else None
        ),
    )


class Attention(TokenMixerBase[AttentionConfig, KVCacheLayer, Float[Array, "suffix_tokens channels"] | QKVInputs]):
    qkvg_projection: Linear
    out_projection: Linear

    query_norm: Normalization | None
    key_norm: Normalization | None

    sinks: Float[Array, " heads"] | None
    relative_profiles: Float[Array, "relative_channels distances"] | None = None
    key_value_projections: tuple[Linear, ...] = ()
    key_conv: SeparableCausalConv | None = None
    value_conv: SeparableCausalConv | None = None

    @property
    def model_dim(self) -> int:
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
        return self.config.has_sinks

    @eqx.filter_jit
    def _prepare_heads(
        self,
        projection: Float[Array, "tokens channels"],
        num_heads: int,
        norm: Normalization | None,
        clip_threshold: float | None,
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
        if clip_threshold is not None:
            heads = clip_norm(heads, clip_threshold).astype(heads.dtype)
        if positional_embeddings is not None:
            heads = call_vmapped(positional_embeddings.apply, heads, in_axes=1, out_axes=1)
        return heads

    def prepare_query_heads(
        self,
        projection: Float[Array, "tokens query_channels"],
        positional_embeddings: PositionalEmbeddings | None,
    ) -> Float[Array, "tokens heads head_channels"]:
        return self._prepare_heads(
            projection,
            self.config.num_heads,
            self.query_norm,
            self.config.query_clip_threshold,
            positional_embeddings,
        )

    def prepare_key_value_heads(
        self,
        key_projection: Float[Array, "tokens key_value_channels"],
        value_projection: Float[Array, "tokens key_value_channels"],
        positional_embeddings: PositionalEmbeddings | None,
    ) -> tuple[Float[Array, "tokens groups head_channels"], Float[Array, "tokens groups head_channels"]]:
        keys = self._prepare_heads(
            key_projection,
            self.config.num_groups,
            self.key_norm,
            self.config.key_clip_threshold,
            positional_embeddings,
        )
        values = rearrange(
            value_projection,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_groups,
            head_channels=self.config.head_dim,
        )
        if self.config.normalize_values:
            values = _rms_normalize(values, eps=1e-6)
        return keys, values

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
        projections = call_vmapped(
            self.qkvg_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        queries = self.prepare_query_heads(projections[0], positional_embeddings)
        gate = projections[-1] if self.config.has_gate else None
        if self.config.is_kv_sharing:
            return queries, None, None, None, gate
        qkvi_projections = projections[:-1] if self.config.has_gate else projections
        _, keys, values, *indexer = qkvi_projections
        indexer_projection = None
        if indexer:
            (indexer_projection,) = indexer
        keys, values = self.prepare_key_value_heads(keys, values, positional_embeddings)
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

    def project_stacked(
        self,
        stacked_config: StackedProjectionConfig,
        projection_index: int,
        inputs: Float[Array, "suffix_tokens channels"],
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "suffix_tokens _"], ...]:
        parts = call_vmapped(
            (self.qkvg_projection, *self.key_value_projections)[projection_index],
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        logical_dims = stacked_config.logical_output_dims(
            self.config.num_heads,
            self.config.num_groups,
            self.config.head_dim,
        )
        return tuple(part[:, :rows] for part, rows in zip(parts, logical_dims[projection_index], strict=True))

    def convolve(
        self,
        conv: SeparableCausalConv,
        projection: KeyValueProjection,
        length_without_padding: Int[Array, ""] | int | None,
        state: Float[Array, "tail_tokens channels"] | None,
        return_updated_state: bool,
    ) -> CausalConvResult:
        return conv(
            projection.states.astype(jnp.float32),
            length_without_padding,
            state,
            return_updated_state,
            coefficient_deltas=(
                rearrange(
                    projection.taps.astype(jnp.float32),
                    "tokens (groups kernel) -> tokens kernel groups",
                    groups=self.config.num_groups,
                )
                if projection.taps is not None
                else None
            ),
        )

    def project_stacked_heads(
        self,
        stacked_config: StackedProjectionConfig,
        inputs: QKVInputs,
        conv_states: KeyValueConvState | None,
        length_without_padding: Int[Array, ""] | int | None,
        return_updated_state: bool,
        positional_embeddings: PositionalEmbeddings | None,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> ProjectedHeads:
        if self.key_conv is None or self.value_conv is None:
            raise ValueError("Stacked attention projections expect key/value convolutions.")
        query_keychain, key_keychain, value_keychain = keychain.split(3)
        match stacked_config.layout:
            case ProjectionLayout.STACKED_QUERY_FIRST:
                query_stack = self.project_stacked(
                    stacked_config, 0, inputs.query, forward_pass_config, keychain=query_keychain
                )
                key_stack = (
                    self.project_stacked(stacked_config, 0, inputs.key, forward_pass_config, keychain=key_keychain)
                    if inputs.key is not None
                    else query_stack
                )
                value_stack = (
                    self.project_stacked(stacked_config, 0, inputs.value, forward_pass_config, keychain=value_keychain)
                    if inputs.value is not None
                    else query_stack
                )
                query_projection, gate, relative_states, sink_logits, _, _ = query_stack
                _, _, _, _, key_states, _ = key_stack
                *_, value_states = value_stack
                key_projection = KeyValueProjection(key_states, None)
                value_projection = KeyValueProjection(value_states, None)
            case ProjectionLayout.GROUPED_WITH_CONV_TAPS:
                query_projection, gate, relative_states, sink_logits = self.project_stacked(
                    stacked_config,
                    0,
                    inputs.query,
                    forward_pass_config,
                    keychain=query_keychain,
                )
                key_states, key_taps = self.project_stacked(
                    stacked_config,
                    1,
                    inputs.key if inputs.key is not None else inputs.query,
                    forward_pass_config,
                    keychain=key_keychain,
                )
                value_states, value_taps = self.project_stacked(
                    stacked_config,
                    2,
                    inputs.value if inputs.value is not None else inputs.query,
                    forward_pass_config,
                    keychain=value_keychain,
                )
                key_projection = KeyValueProjection(key_states, key_taps)
                value_projection = KeyValueProjection(value_states, value_taps)
        key_conv = self.convolve(
            self.key_conv,
            key_projection,
            length_without_padding,
            conv_states.keys if conv_states is not None else None,
            return_updated_state,
        )
        value_conv = self.convolve(
            self.value_conv,
            value_projection,
            length_without_padding,
            conv_states.values if conv_states is not None else None,
            return_updated_state,
        )
        keys, values = self.prepare_key_value_heads(
            key_conv.outputs.astype(query_projection.dtype),
            value_conv.outputs.astype(query_projection.dtype),
            positional_embeddings,
        )
        return ProjectedHeads(
            queries=self.prepare_query_heads(query_projection, positional_embeddings),
            keys=keys,
            values=values,
            gate=gate,
            relative_states=relative_states,
            sink_logits=sink_logits,
            conv_states=(
                KeyValueConvState(key_conv.state, value_conv.state)
                if key_conv.state is not None and value_conv.state is not None
                else None
            ),
        )

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"] | QKVInputs,
        positional_embeddings: PositionalEmbeddings | None,
        state: KVCacheLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        reuse_cache: bool = False,
        *,
        keychain: Keychain,
        sparse_mask: SparseMask | None = None,
    ) -> AttentionResult:
        qkvg_keychain, out_keychain = keychain.split(2)
        stacked_config = self.config.stacked_projection_config
        if stacked_config is None:
            if isinstance(inputs, QKVInputs):
                raise TypeError("Fused attention projections take a single input.")
            queries, keys, values, _, gate = self.project_heads(
                inputs, positional_embeddings, forward_pass_config, keychain=qkvg_keychain
            )
            heads = ProjectedHeads(queries, keys, values, gate, None, None, None)
        else:
            heads = self.project_stacked_heads(
                stacked_config,
                inputs if isinstance(inputs, QKVInputs) else QKVInputs(query=inputs, key=None, value=None),
                state.conv_states if state is not None else None,
                length_without_padding,
                return_updated_state,
                positional_embeddings,
                forward_pass_config,
                keychain=qkvg_keychain,
            )
        result = self.attend(
            heads.queries,
            heads.keys,
            heads.values,
            heads.gate,
            state,
            return_updated_state,
            length_without_padding,
            forward_pass_config,
            attention_parent_indices,
            reuse_cache,
            keychain=out_keychain,
            sparse_mask=sparse_mask,
            relative_states=heads.relative_states,
            sink_logits=heads.sink_logits,
        )
        if result.state is None or heads.conv_states is None:
            return result
        return AttentionResult(outputs=result.outputs, state=replace(result.state, conv_states=heads.conv_states))

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
        relative_states: Float[Array, "tokens relative_channels"] | None = None,
        sink_logits: Float[Array, "tokens heads"] | None = None,
    ) -> AttentionResult:
        if sparse_mask is not None and attention_parent_indices is not None:
            raise ValueError("A sparse mask cannot be combined with tree attention.")
        if sparse_mask is not None and (self.has_sinks or self.relative_profiles is not None):
            raise ValueError("A sparse mask cannot be combined with attention sinks or relative biases.")
        if attention_parent_indices is not None and (
            self.relative_profiles is not None or self.config.log_scaling_config is not None
        ):
            raise ValueError("Tree attention does not support position-dependent attention logits.")
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

        query_positions, key_positions = updated_state.query_key_positions(num_suffix_tokens, length_without_padding)
        if self.config.log_scaling_config is not None:
            position_scales = self.config.log_scaling_config.scales(query_positions - int(updated_state.has_sinks))
            queries = queries.astype(jnp.float32) * position_scales[:, None, None]
        else:
            position_scales = None
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
            bias = self.attention_bias(
                relative_states,
                sink_logits,
                query_positions,
                key_positions,
                position_scales,
            )

            attention_output = _attention_kernel(
                queries,
                updated_state.keys,
                updated_state.values,
                bias=bias.astype(queries.dtype) if bias is not None else None,
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

    def relative_bias(
        self,
        relative_states: Float[Array, "dst_tokens relative_channels"] | None,
        query_positions: Int[Array, " dst_tokens"],
        key_positions: Int[Array, " src_tokens"],
        position_scales: Float[Array, " dst_tokens"] | None,
    ) -> Float[Array, "heads dst_tokens src_tokens"] | None:
        stacked_config = self.config.stacked_projection_config
        if stacked_config is None or self.relative_profiles is None:
            if relative_states is not None:
                raise ValueError("Relative states were given to an attention without a relative bias.")
            return None
        if relative_states is None:
            raise ValueError("An attention with a relative bias requires relative states.")
        relative_bias_config = stacked_config.relative_bias_config
        scaled_states = clip_norm(
            rearrange(
                relative_states, "tokens (heads channels) -> tokens heads channels", heads=self.config.num_heads
            ),
            relative_bias_config.clip_threshold,
        )
        if position_scales is not None:
            scaled_states = scaled_states * position_scales[:, None, None]
        distances = query_positions[:, None] - key_positions[None, :]
        relative_logits = einsum(
            scaled_states,
            self.relative_profiles.astype(jnp.float32),
            "tokens heads channels, channels distances -> tokens heads distances",
        )
        relative_bias = jnp.take_along_axis(
            relative_logits,
            jnp.clip(distances, 0, relative_bias_config.extent - 1)[:, None, :],
            axis=2,
        )
        in_extent = (distances >= 0) & (distances < relative_bias_config.extent)
        return rearrange(
            jnp.where(in_extent[:, None, :], relative_bias, 0.0),
            "tokens heads sources -> heads tokens sources",
        )

    def attention_bias(
        self,
        relative_states: Float[Array, "dst_tokens relative_channels"] | None,
        sink_logits: Float[Array, "dst_tokens heads"] | None,
        query_positions: Int[Array, " dst_tokens"],
        key_positions: Int[Array, " src_tokens"],
        position_scales: Float[Array, " dst_tokens"] | None,
    ) -> Float[Array, "heads dst_tokens src_tokens"] | None:
        if (sink_logits is not None) != (self.config.stacked_projection_config is not None):
            raise ValueError("Sink logits must be given exactly when the attention projects its sinks.")
        (num_queries,) = query_positions.shape
        (num_keys,) = key_positions.shape
        if self.sinks is not None:
            sink_logits = jnp.broadcast_to(self.sinks, (num_queries, self.config.num_heads))
        relative_bias = self.relative_bias(relative_states, query_positions, key_positions, position_scales)
        if sink_logits is None:
            return relative_bias
        if relative_bias is None:
            relative_bias = jnp.zeros((self.config.num_heads, num_queries, num_keys), dtype=jnp.float32)
        return relative_bias.at[:, :, 0].set(
            rearrange(sink_logits.astype(jnp.float32), "tokens heads -> heads tokens"),
        )

    def paged_decode(
        self,
        inputs: Float[Array, "batch 1 channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: PagedKVCacheLayer,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[PagedKVCacheLayer]:
        if not self.config.supports_paged_decode:
            raise ValueError("Paged decoding supports neither stacked projection layouts nor log scaling.")
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

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StaticKVCacheLayer:
        cache = StaticKVCacheLayer.init(
            self.has_sinks,
            capacity,
            self.config.num_groups,
            self.config.head_dim,
            dtype,
            self.sharding_config,
        )
        if self.key_conv is None or self.value_conv is None:
            return cache
        key_conv_state, value_conv_state = (
            jnp.zeros((conv.kernel_size - 1, conv.input_dim), dtype=jnp.float32)
            for conv in (self.key_conv, self.value_conv)
        )
        return replace(cache, conv_states=KeyValueConvState(key_conv_state, value_conv_state))
