from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.kernels.attention import pallas_decode_attention, stable_reduction_attention, xla_attention
from lalamo.module import Keychain
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

from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, StaticKVCacheLayer

__all__ = [
    "Attention",
    "AttentionBase",
    "AttentionConfig",
    "AttentionConfigBase",
    "AttentionParameters",
    "AttentionResult",
    "OwnedKeyValueProjection",
    "SharedKeyValueProjection",
]


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


AttentionResult = TokenMixerResult[KVCacheLayer]


class AttentionParameters(NamedTuple):
    qkvg_projection: Linear
    out_projection: Linear
    query_norm: Normalization | None
    key_norm: Normalization | None
    sinks: Float[Array, " heads"] | None


@dataclass(frozen=True)
class AttentionConfigBase(TokenMixerConfig):
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

    def init_parameters(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> AttentionParameters:
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

        qkvg_projection = self.qkvg_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=output_dims,
            has_biases=self.has_qkvg_biases,
        )
        out_projection = self.out_projection_config.init(
            initializer,
            self.num_heads * self.head_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
        )

        if self.query_norm_config is not None:
            query_norm = self.query_norm_config.init(
                initializer,
                input_dim=self.head_dim,
            )
        else:
            query_norm = None

        if self.key_norm_config is not None and not self.is_kv_sharing:
            key_norm = self.key_norm_config.init(
                initializer,
                input_dim=self.head_dim,
            )
        else:
            key_norm = None

        if self.has_sinks:
            sinks = initializer.zeros((self.num_heads,))
        else:
            sinks = None

        return AttentionParameters(
            qkvg_projection=qkvg_projection,
            out_projection=out_projection,
            query_norm=query_norm,
            key_norm=key_norm,
            sinks=sinks,
        )


@dataclass(frozen=True)
class AttentionConfig(AttentionConfigBase):
    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "Attention":
        parameters = self.init_parameters(initializer, model_dim)
        return Attention(
            config=self,
            sharding_config=initializer.sharding_config,
            qkvg_projection=parameters.qkvg_projection,
            out_projection=parameters.out_projection,
            query_norm=parameters.query_norm,
            key_norm=parameters.key_norm,
            sinks=parameters.sinks,
        )


class OwnedKeyValueProjection(NamedTuple):
    queries: Float[Array, "tokens heads head_channels"]
    gate: Float[Array, "tokens gate_channels"] | None
    keys: Float[Array, "tokens groups head_channels"]
    values: Float[Array, "tokens groups head_channels"]


class SharedKeyValueProjection(NamedTuple):
    queries: Float[Array, "tokens heads head_channels"]
    gate: Float[Array, "tokens gate_channels"] | None


class AttentionBase[ConfigT: AttentionConfigBase, StateLayerT: StateLayerBase](TokenMixerBase[ConfigT, StateLayerT]):
    qkvg_projection: Linear
    out_projection: Linear

    query_norm: Normalization | None
    key_norm: Normalization | None

    sinks: Float[Array, " heads"] | None

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

    def project(
        self,
        inputs: Float[Array, "tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> OwnedKeyValueProjection | SharedKeyValueProjection:
        projections = call_vmapped(
            self.qkvg_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        queries = self._prepare_heads(projections[0], self.config.num_heads, self.query_norm, positional_embeddings)
        if self.config.has_gate:
            gate = projections[-1]
        else:
            gate = None
        if self.config.is_kv_sharing:
            return SharedKeyValueProjection(queries=queries, gate=gate)

        _, keys, values, *_ = projections
        keys = self._prepare_heads(keys, self.config.num_groups, self.key_norm, positional_embeddings)
        values = rearrange(
            values,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_groups,
            head_channels=self.config.head_dim,
        )
        if self.config.normalize_values:
            values = _rms_normalize(values, eps=1e-6)
        return OwnedKeyValueProjection(queries=queries, gate=gate, keys=keys, values=values)

    def output_projection(
        self,
        attention_output: Float[Array, "tokens heads head_channels"],
        gate: Float[Array, "tokens gate_channels"] | None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "tokens channels"]:
        flattened_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        if gate is not None:
            flattened_output = flattened_output * jax.nn.sigmoid(gate)
        (result,) = call_vmapped(
            self.out_projection,
            flattened_output,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        return result


class Attention(AttentionBase[AttentionConfig, KVCacheLayer]):
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
        projection = self.project(inputs, positional_embeddings, forward_pass_config, keychain=keychain)
        if isinstance(projection, SharedKeyValueProjection):
            raise TypeError("KV-sharing attention layers do not own key/value projections.")
        return projection.keys, projection.values

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: KVCacheLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        reuse_cache: bool = False,
        *,
        keychain: Keychain,
    ) -> AttentionResult:
        qkvg_keychain, out_keychain = keychain.split(2)
        assert reuse_cache == self.config.is_kv_sharing, "reuse_cache must match AttentionConfig.is_kv_sharing"
        projection = self.project(inputs, positional_embeddings, forward_pass_config, keychain=qkvg_keychain)

        num_suffix_tokens, _, _ = projection.queries.shape
        if isinstance(projection, SharedKeyValueProjection):
            if state is None:
                raise ValueError("a KV-sharing layer must receive the source layer's cache as `state`")
            prefix_length = state.current_prefix_length() - num_suffix_tokens
            updated_state = state
        else:
            prefix_length = 0 if state is None else state.current_prefix_length()
            if state is None:
                updated_state = DynamicKVCacheLayer.init(
                    self.has_sinks,
                    projection.keys.astype(projection.values.dtype),
                    projection.values,
                    length=length_without_padding,
                )
            else:
                updated_state = state.extend(projection.keys, projection.values, added_length=length_without_padding)

        queries = projection.queries.astype(updated_state.keys.dtype)
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
            sink_bias = sink_bias.at[:, :, 0].set(self.sinks[:, None])
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
        result = self.output_projection(
            attention_output,
            projection.gate,
            forward_pass_config,
            keychain=out_keychain,
        )

        if not return_updated_state:
            updated_state = None

        return AttentionResult(
            outputs=result,
            state=updated_state,
        )

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StaticKVCacheLayer:
        return StaticKVCacheLayer.init(
            self.has_sinks,
            capacity,
            self.config.num_groups,
            self.config.head_dim,
            dtype,
            self.sharding_config,
        )
