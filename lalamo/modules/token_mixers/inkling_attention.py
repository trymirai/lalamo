from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    QKVInputs,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

from .attention import attention_kernel
from .convolutions import CausalConvResult, SeparableCausalConv, SeparableCausalConvConfig
from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, StaticKVCacheLayer

__all__ = [
    "InklingAttention",
    "InklingAttentionConfig",
    "InklingAttentionState",
    "KeyValueLayout",
    "LogScalingConfig",
    "aligned_rows",
]


def aligned_rows(rows: int, alignment: int) -> int:
    return -(-rows // alignment) * alignment


def clip_norm(heads: Float[Array, "*dims channels"], threshold: float) -> Float[Array, "*dims channels"]:
    upcast_heads = heads.astype(jnp.float32)
    norms = jnp.linalg.norm(upcast_heads, axis=-1, keepdims=True)
    return upcast_heads * jnp.minimum(threshold / (norms + 1e-6), 1.0)


class KeyValueLayout(StrEnum):
    STACKED_WITH_QUERY = "stacked_with_query"
    GROUPED_WITH_CONV_TAPS = "grouped_with_conv_taps"


@dataclass(frozen=True)
class LogScalingConfig(LalamoConfig):
    alpha: float
    position_floor: int

    def scales(self, token_positions: Int[Array, " tokens"]) -> Float[Array, " tokens"]:
        relative_positions = (token_positions + 1).astype(jnp.float32) / self.position_floor
        return 1 + self.alpha * jnp.log(jnp.maximum(relative_positions, 1.0))


class InklingAttentionState(StateLayerBase):
    kv_cache: KVCacheLayer
    key_conv: Float[Array, "*batch tokens channels"]
    value_conv: Float[Array, "*batch tokens channels"]


@dataclass(frozen=True)
class InklingAttentionConfig(TokenMixerConfig):
    projection_config: LinearConfig
    norm_config: NormalizationConfig
    conv_config: SeparableCausalConvConfig
    log_scaling_config: LogScalingConfig | None
    key_value_layout: KeyValueLayout

    num_heads: int
    num_groups: int
    head_dim: int
    relative_dim: int
    relative_extent: int
    conv_kernel_size: int
    row_alignment: int
    sliding_window_size: int | None
    attention_scale: float
    query_clip_threshold: float
    key_clip_threshold: float
    relative_clip_threshold: float

    def __post_init__(self) -> None:
        if self.norm_config.has_scale:
            raise ValueError("Inkling attention expects gamma-free norms; fold the gammas into the projections.")

    @property
    def query_rows(self) -> tuple[int, ...]:
        query_dim = self.num_heads * self.head_dim
        return (query_dim, query_dim, self.num_heads * self.relative_dim, self.num_heads)

    @property
    def key_value_rows(self) -> tuple[int, ...]:
        match self.key_value_layout:
            case KeyValueLayout.STACKED_WITH_QUERY:
                return (self.num_groups * self.head_dim,)
            case KeyValueLayout.GROUPED_WITH_CONV_TAPS:
                return (self.num_groups * self.head_dim, self.num_groups * self.conv_kernel_size)

    @property
    def projection_rows(self) -> tuple[tuple[int, ...], ...]:
        match self.key_value_layout:
            case KeyValueLayout.STACKED_WITH_QUERY:
                return ((*self.query_rows, *self.key_value_rows, *self.key_value_rows),)
            case KeyValueLayout.GROUPED_WITH_CONV_TAPS:
                return (self.query_rows, self.key_value_rows, self.key_value_rows)

    def init(self, initializer: Initializer, model_dim: int) -> "InklingAttention":
        key_value_dim = self.num_groups * self.head_dim
        return InklingAttention(
            config=self,
            sharding_config=initializer.sharding_config,
            norm=self.norm_config.init(initializer, model_dim),
            projections=tuple(
                self.projection_config.init(
                    initializer,
                    input_dim=model_dim,
                    output_dims=tuple(aligned_rows(part, self.row_alignment) for part in rows),
                    has_biases=False,
                )
                for rows in self.projection_rows
            ),
            out_projection=self.projection_config.init(
                initializer,
                input_dim=self.num_heads * self.head_dim,
                output_dims=(model_dim,),
                has_biases=False,
            ),
            relative_profiles=initializer.zeros((self.relative_dim, self.relative_extent), dtype=jnp.float32),
            key_conv=self.conv_config.init(initializer, key_value_dim, self.conv_kernel_size, dtype=jnp.float32),
            value_conv=self.conv_config.init(initializer, key_value_dim, self.conv_kernel_size, dtype=jnp.float32),
        )


class InklingAttention(TokenMixerBase[InklingAttentionConfig, InklingAttentionState, QKVInputs]):
    norm: Normalization
    projections: tuple[Linear, ...]
    out_projection: Linear
    relative_profiles: Float[Array, "relative_channels distances"]
    key_conv: SeparableCausalConv
    value_conv: SeparableCausalConv

    @property
    def model_dim(self) -> int:
        return self.projections[0].input_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return PositionalEmbeddingSelector.NONE

    def project(
        self,
        projection_index: int,
        inputs: Float[Array, "suffix_tokens channels"],
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "suffix_tokens _"], ...]:
        parts = call_vmapped(
            self.projections[projection_index],
            call_vmapped(self.norm, inputs),
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        return tuple(
            part[:, :rows] for part, rows in zip(parts, self.config.projection_rows[projection_index], strict=True)
        )

    def project_heads(
        self,
        inputs: QKVInputs,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> tuple[tuple[Float[Array, "suffix_tokens _"], ...], ...]:
        query_keychain, key_keychain, value_keychain = keychain.split(3)
        match self.config.key_value_layout:
            case KeyValueLayout.STACKED_WITH_QUERY:
                query_parts_end = len(self.config.query_rows)
                key_parts_end = query_parts_end + len(self.config.key_value_rows)
                query_stack = self.project(0, inputs.query, forward_pass_config, keychain=query_keychain)
                key_stack = (
                    self.project(0, inputs.key, forward_pass_config, keychain=key_keychain)
                    if inputs.key is not None
                    else query_stack
                )
                value_stack = (
                    self.project(0, inputs.value, forward_pass_config, keychain=value_keychain)
                    if inputs.value is not None
                    else query_stack
                )
                return (
                    query_stack[:query_parts_end],
                    key_stack[query_parts_end:key_parts_end],
                    value_stack[key_parts_end:],
                )
            case KeyValueLayout.GROUPED_WITH_CONV_TAPS:
                return (
                    self.project(0, inputs.query, forward_pass_config, keychain=query_keychain),
                    self.project(
                        1,
                        inputs.key if inputs.key is not None else inputs.query,
                        forward_pass_config,
                        keychain=key_keychain,
                    ),
                    self.project(
                        2,
                        inputs.value if inputs.value is not None else inputs.query,
                        forward_pass_config,
                        keychain=value_keychain,
                    ),
                )

    def convolve(
        self,
        conv: SeparableCausalConv,
        projections: tuple[Float[Array, "suffix_tokens _"], ...],
        length_without_padding: Int[Array, ""] | int | None,
        state: Float[Array, "tokens channels"] | None,
        return_updated_state: bool,
    ) -> CausalConvResult:
        match projections:
            case (states,):
                return conv(states.astype(jnp.float32), length_without_padding, state, return_updated_state)
            case (states, taps):
                return conv(
                    states.astype(jnp.float32),
                    length_without_padding,
                    state,
                    return_updated_state,
                    coefficient_deltas=rearrange(
                        taps.astype(jnp.float32),
                        "tokens (groups kernel) -> tokens kernel groups",
                        groups=self.config.num_groups,
                    ),
                )
        raise ValueError(f"Expected one or two key/value projections, got {len(projections)}.")

    def __call__(
        self,
        inputs: QKVInputs,
        positional_embeddings: PositionalEmbeddings | None,
        state: InklingAttentionState | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        reuse_cache: bool = False,
        token_positions: Int[Array, " suffix_tokens"] | None = None,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[InklingAttentionState]:
        if positional_embeddings is not None:
            raise ValueError("Inkling attention does not use rotary embeddings.")
        if attention_parent_indices is not None:
            raise ValueError("Inkling attention does not support tree attention.")
        if reuse_cache:
            raise ValueError("Inkling attention does not support KV cache sharing.")
        if token_positions is None:
            raise ValueError("Inkling attention requires token positions.")
        projection_keychain, out_keychain = keychain.split()
        query_parts, key_parts, value_parts = self.project_heads(
            inputs,
            forward_pass_config,
            keychain=projection_keychain,
        )
        query_heads, gate, relative_states, sink_logits = query_parts
        key_conv = self.convolve(
            self.key_conv,
            key_parts,
            length_without_padding,
            state.key_conv if state is not None else None,
            return_updated_state,
        )
        value_conv = self.convolve(
            self.value_conv,
            value_parts,
            length_without_padding,
            state.value_conv if state is not None else None,
            return_updated_state,
        )

        queries = clip_norm(
            rearrange(query_heads, "tokens (heads channels) -> tokens heads channels", heads=self.config.num_heads),
            self.config.query_clip_threshold,
        )
        relative_states = clip_norm(
            rearrange(
                relative_states, "tokens (heads channels) -> tokens heads channels", heads=self.config.num_heads
            ),
            self.config.relative_clip_threshold,
        )
        if self.config.log_scaling_config is not None:
            position_scales = self.config.log_scaling_config.scales(token_positions)[:, None, None]
            queries = queries * position_scales
            relative_states = relative_states * position_scales
        keys = clip_norm(
            rearrange(
                key_conv.outputs, "tokens (groups channels) -> tokens groups channels", groups=self.config.num_groups
            ),
            self.config.key_clip_threshold,
        )
        values = call_vmapped_twice(
            self.norm,
            rearrange(
                value_conv.outputs, "tokens (groups channels) -> tokens groups channels", groups=self.config.num_groups
            ),
        )

        num_suffix_tokens, _, _ = queries.shape
        if state is None:
            kv_cache = DynamicKVCacheLayer.init(
                has_sinks=True,
                keys=keys.astype(values.dtype),
                values=values,
                length=length_without_padding,
            )
        else:
            kv_cache = state.kv_cache.extend(keys, values, added_length=length_without_padding)
        query_positions, key_positions = kv_cache.query_key_positions(num_suffix_tokens, length_without_padding)
        distances = query_positions[:, None] - key_positions[None, :]
        relative_logits = einsum(
            relative_states,
            self.relative_profiles.astype(jnp.float32),
            "tokens heads channels, channels distances -> tokens heads distances",
        )
        relative_bias = jnp.take_along_axis(
            relative_logits,
            jnp.broadcast_to(
                jnp.clip(distances, 0, self.config.relative_extent - 1)[:, None, :],
                (num_suffix_tokens, self.config.num_heads, distances.shape[1]),
            ),
            axis=2,
        )
        in_extent = (distances >= 0) & (distances < self.config.relative_extent)
        bias = rearrange(
            jnp.where(in_extent[:, None, :], relative_bias, 0.0),
            "tokens heads sources -> heads tokens sources",
        )
        bias = bias.at[:, :, 0].set(rearrange(sink_logits.astype(jnp.float32), "tokens heads -> heads tokens"))

        attention_output = attention_kernel(
            queries.astype(kv_cache.keys.dtype),
            kv_cache.keys,
            kv_cache.values,
            bias=bias.astype(kv_cache.keys.dtype),
            mask=kv_cache.attention_mask(
                num_suffix_tokens,
                is_causal=True,
                suffix_length_without_padding=length_without_padding,
                sliding_window_size=self.config.sliding_window_size,
            ),
            scale=self.config.attention_scale,
            logit_soft_cap=None,
            forward_pass_config=forward_pass_config,
        )
        gated_output = rearrange(attention_output, "tokens heads channels -> tokens (heads channels)").astype(
            jnp.float32,
        ) * jax.nn.sigmoid(gate.astype(jnp.float32))
        (outputs,) = call_vmapped(
            self.out_projection,
            gated_output.astype(attention_output.dtype),
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain,
        )
        if not return_updated_state:
            return TokenMixerResult(outputs)
        assert key_conv.state is not None and value_conv.state is not None
        return TokenMixerResult(
            outputs,
            InklingAttentionState(kv_cache=kv_cache, key_conv=key_conv.state, value_conv=value_conv.state),
        )

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> InklingAttentionState:
        conv_state_shape = (self.config.conv_kernel_size - 1, self.config.num_groups * self.config.head_dim)
        return InklingAttentionState(
            kv_cache=StaticKVCacheLayer.init(
                has_sinks=True,
                capacity=capacity,
                num_groups=self.config.num_groups,
                head_dim=self.config.head_dim,
                dtype=dtype,
                sharding_config=self.sharding_config,
            ),
            key_conv=jnp.zeros(conv_state_shape, dtype=jnp.float32),
            value_conv=jnp.zeros(conv_state_shape, dtype=jnp.float32),
        )
