from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.modules.activations import Activation
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.mlp import DenseMLP, DenseMLPConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig, UpcastMode
from lalamo.modules.rope import PositionalEmbeddings, RoPE, RoPEConfig
from lalamo.modules.token_mixers.attention import _attention_kernel
from lalamo.modules.transformer_layer import TransformerForwardPassConfig
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

__all__ = [
    "DFlashAttention",
    "DFlashAttentionConfig",
    "DFlashDraftConfig",
    "DFlashDraftLayer",
    "DFlashDraftLayerConfig",
    "DFlashDraftModel",
    "DFlashDraftResult",
    "make_qwen3_dflash_draft_config",
]


@dataclass(frozen=True)
class DFlashAttentionConfig(LalamoConfig):
    linear_config: LinearConfig
    query_norm_config: NormalizationConfig
    key_norm_config: NormalizationConfig
    rope_config: RoPEConfig
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    has_attention_biases: bool
    has_output_biases: bool
    sliding_window_size: int | None
    scale: float

    def init(self, initializer: Initializer, model_dim: int) -> "DFlashAttention":
        query_dim = self.num_heads * self.head_dim
        key_value_dim = self.num_key_value_heads * self.head_dim
        return DFlashAttention(
            config=self,
            sharding_config=initializer.sharding_config,
            query_projection=self.linear_config.init(
                initializer,
                model_dim,
                (query_dim,),
                has_biases=self.has_attention_biases,
            ),
            key_value_projection=self.linear_config.init(
                initializer,
                model_dim,
                (key_value_dim, key_value_dim),
                has_biases=self.has_attention_biases,
            ),
            output_projection=self.linear_config.init(
                initializer,
                query_dim,
                (model_dim,),
                has_biases=self.has_output_biases,
            ),
            query_norm=self.query_norm_config.init(initializer, self.head_dim),
            key_norm=self.key_norm_config.init(initializer, self.head_dim),
            rope=self.rope_config.init(initializer),
        )


class DFlashAttention(LalamoModule[DFlashAttentionConfig]):
    query_projection: Linear
    key_value_projection: Linear
    output_projection: Linear
    query_norm: Normalization
    key_norm: Normalization
    rope: RoPE

    @property
    def model_dim(self) -> int:
        return self.query_projection.input_dim

    def _reshape_queries(
        self,
        queries: Float[Array, "tokens channels"],
    ) -> Float[Array, "tokens heads head_channels"]:
        return rearrange(
            queries,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )

    def _reshape_keys_or_values(
        self,
        keys_or_values: Float[Array, "tokens channels"],
    ) -> Float[Array, "tokens groups head_channels"]:
        return rearrange(
            keys_or_values,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_key_value_heads,
            head_channels=self.config.head_dim,
        )

    def _apply_rope(
        self,
        heads: Float[Array, "tokens heads head_channels"],
        positional_embeddings: PositionalEmbeddings,
    ) -> Float[Array, "tokens heads head_channels"]:
        return call_vmapped(positional_embeddings.apply, heads, in_axes=1, out_axes=1)

    def _normalize_heads(
        self,
        norm: Normalization,
        heads: Float[Array, "tokens heads head_channels"],
        forward_pass_config: TransformerForwardPassConfig,
    ) -> Float[Array, "tokens heads head_channels"]:
        return call_vmapped_twice(
            norm,
            heads,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
        )

    def _attention_mask(
        self,
        attention_mask: Bool[Array, "query_tokens key_tokens"] | None,
        num_query_tokens: int,
        num_key_tokens: int,
    ) -> Bool[Array, "query_tokens key_tokens"] | None:
        if self.config.sliding_window_size is None:
            return attention_mask
        key_indices = jnp.arange(num_key_tokens, dtype=jnp.int32)
        query_indices = jnp.arange(num_query_tokens, dtype=jnp.int32) + num_key_tokens - num_query_tokens
        window_radius = self.config.sliding_window_size // 2
        sliding_window_mask = (key_indices[None, :] >= query_indices[:, None] - window_radius) & (
            key_indices[None, :] <= query_indices[:, None] + window_radius
        )
        if attention_mask is None:
            return sliding_window_mask
        return attention_mask & sliding_window_mask

    @eqx.filter_jit
    def call_unbatched(
        self,
        hidden_states: Float[Array, "query_tokens channels"],
        target_hidden: Float[Array, "context_tokens channels"],
        token_positions: Int[Array, " key_tokens"],
        attention_mask: Bool[Array, "query_tokens key_tokens"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "query_tokens channels"]:
        if hidden_states.ndim != 2:
            raise ValueError(f"hidden_states must be 2D, got shape {hidden_states.shape}")
        if target_hidden.ndim != 2:
            raise ValueError(f"target_hidden must be 2D, got shape {target_hidden.shape}")

        query_keychain, key_value_keychain, output_keychain = keychain.split(3)
        num_query_tokens, _ = hidden_states.shape
        source_states = jnp.concatenate([target_hidden, hidden_states], axis=0)
        num_key_tokens, _ = source_states.shape
        if token_positions.shape != (num_key_tokens,):
            raise ValueError(
                "token_positions must have one position per source token:"
                f" expected {(num_key_tokens,)}, got {token_positions.shape}",
            )

        (query_projection,) = call_vmapped(
            self.query_projection,
            hidden_states,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config.matmul_config,
            keychain=query_keychain,
        )
        key_projection, value_projection = call_vmapped(
            self.key_value_projection,
            source_states,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config.matmul_config,
            keychain=key_value_keychain,
        )

        queries = self._reshape_queries(query_projection)
        keys = self._reshape_keys_or_values(key_projection)
        values = self._reshape_keys_or_values(value_projection)
        queries = self._normalize_heads(self.query_norm, queries, forward_pass_config)
        keys = self._normalize_heads(self.key_norm, keys, forward_pass_config)

        positional_embeddings = self.rope(token_positions).astype(
            forward_pass_config.mixer_forward_pass_config.rope_dtype,
        )
        query_positional_embeddings = PositionalEmbeddings(
            cosines=positional_embeddings.cosines[-num_query_tokens:],
            sines=positional_embeddings.sines[-num_query_tokens:],
        )
        queries = self._apply_rope(queries, query_positional_embeddings)
        keys = self._apply_rope(keys, positional_embeddings)

        effective_mask = self._attention_mask(attention_mask, num_query_tokens, num_key_tokens)
        attention_output = _attention_kernel(
            queries,
            keys,
            values,
            bias=None,
            mask=effective_mask,
            scale=self.config.scale,
            logit_soft_cap=None,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config,
        )
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        (result,) = call_vmapped(
            self.output_projection,
            attention_output,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config.matmul_config,
            keychain=output_keychain,
        )
        return result

    @eqx.filter_jit
    def __call__(
        self,
        hidden_states: Float[Array, "batch query_tokens channels"],
        target_hidden: Float[Array, "batch context_tokens channels"],
        token_positions: Int[Array, "batch key_tokens"],
        attention_mask: Bool[Array, "batch query_tokens key_tokens"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch query_tokens channels"]:
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        if attention_mask is None:
            return call_vmapped(
                self.call_unbatched,
                hidden_states,
                target_hidden,
                token_positions,
                forward_pass_config=forward_pass_config,
                keychain=keychain,
                added_sharding_axis=batch_axis,
            )
        return call_vmapped(
            self.call_unbatched,
            hidden_states,
            target_hidden,
            token_positions,
            attention_mask,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            added_sharding_axis=batch_axis,
        )


@dataclass(frozen=True)
class DFlashDraftLayerConfig(LalamoConfig):
    attention_config: DFlashAttentionConfig
    input_norm_config: NormalizationConfig
    post_attention_norm_config: NormalizationConfig
    mlp_config: DenseMLPConfig

    def init(self, initializer: Initializer, model_dim: int, hidden_dim: int) -> "DFlashDraftLayer":
        return DFlashDraftLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            attention=self.attention_config.init(initializer, model_dim),
            input_norm=self.input_norm_config.init(initializer, model_dim),
            post_attention_norm=self.post_attention_norm_config.init(initializer, model_dim),
            mlp=self.mlp_config.init(initializer, model_dim, hidden_dim),
        )


class DFlashDraftLayer(LalamoModule[DFlashDraftLayerConfig]):
    attention: DFlashAttention
    input_norm: Normalization
    post_attention_norm: Normalization
    mlp: DenseMLP

    @eqx.filter_jit
    def __call__(
        self,
        hidden_states: Float[Array, "batch query_tokens channels"],
        target_hidden: Float[Array, "batch context_tokens channels"],
        token_positions: Int[Array, "batch key_tokens"],
        attention_mask: Bool[Array, "batch query_tokens key_tokens"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch query_tokens channels"]:
        attention_keychain, mlp_keychain = keychain.split(2)
        normalized_attention_inputs = call_vmapped_twice(
            self.input_norm,
            hidden_states,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        attention_outputs = self.attention(
            normalized_attention_inputs,
            target_hidden,
            token_positions,
            attention_mask,
            forward_pass_config=forward_pass_config,
            keychain=attention_keychain,
        )
        mlp_inputs = hidden_states + attention_outputs
        normalized_mlp_inputs = call_vmapped_twice(
            self.post_attention_norm,
            mlp_inputs,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        mlp_outputs = self.mlp(
            normalized_mlp_inputs,
            forward_pass_config=forward_pass_config.mlp_forward_pass_config,
            keychain=mlp_keychain,
        )
        return mlp_inputs + mlp_outputs


@dataclass(frozen=True)
class DFlashDraftConfig(LalamoConfig):
    model_dim: int
    hidden_dim: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    vocab_size: int
    context_projection_config: LinearConfig
    context_norm_config: NormalizationConfig
    layer_configs: tuple[DFlashDraftLayerConfig, ...]
    output_norm_config: NormalizationConfig

    def init(self, initializer: Initializer) -> "DFlashDraftModel":
        context_feature_dim = len(self.target_layer_ids) * self.model_dim
        return DFlashDraftModel(
            config=self,
            sharding_config=initializer.sharding_config,
            context_projection=self.context_projection_config.init(
                initializer,
                context_feature_dim,
                (self.model_dim,),
                has_biases=False,
            ),
            context_norm=self.context_norm_config.init(initializer, self.model_dim),
            layers=tuple(
                layer_config.init(initializer, self.model_dim, self.hidden_dim) for layer_config in self.layer_configs
            ),
            output_norm=self.output_norm_config.init(initializer, self.model_dim),
        )


class DFlashDraftResult(Exportable, eqx.Module):
    hidden_states: Float[Array, "batch query_tokens channels"]


class DFlashDraftModel(LalamoModule[DFlashDraftConfig]):
    context_projection: Linear
    context_norm: Normalization
    layers: tuple[DFlashDraftLayer, ...]
    output_norm: Normalization

    @property
    def target_feature_dim(self) -> int:
        return len(self.config.target_layer_ids) * self.config.model_dim

    @eqx.filter_jit
    def __call__(
        self,
        noise_embeddings: Float[Array, "batch query_tokens channels"],
        target_features: Float[Array, "batch context_tokens target_channels"],
        token_positions: Int[Array, "batch key_tokens"],
        attention_mask: Bool[Array, "batch query_tokens key_tokens"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> DFlashDraftResult:
        if noise_embeddings.ndim != 3:
            raise ValueError(f"noise_embeddings must be 3D, got shape {noise_embeddings.shape}")
        if target_features.ndim != 3:
            raise ValueError(f"target_features must be 3D, got shape {target_features.shape}")
        if target_features.shape[-1] != self.target_feature_dim:
            raise ValueError(
                f"target_features must have feature dimension {self.target_feature_dim},"
                f" got {target_features.shape[-1]}",
            )

        context_keychain, *layer_keychains = keychain.split(len(self.layers) + 1)
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        (target_hidden,) = call_vmapped_twice(
            self.context_projection,
            target_features,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config.matmul_config,
            keychain=context_keychain,
            added_sharding_axes=(batch_axis, None),
        )
        target_hidden = call_vmapped_twice(
            self.context_norm,
            target_hidden,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
            added_sharding_axes=(batch_axis, None),
        )

        hidden_states = noise_embeddings
        for layer, layer_keychain in zip(self.layers, layer_keychains, strict=True):
            hidden_states = layer(
                hidden_states,
                target_hidden,
                token_positions,
                attention_mask,
                forward_pass_config=forward_pass_config,
                keychain=layer_keychain,
            )

        hidden_states = call_vmapped_twice(
            self.output_norm,
            hidden_states,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
            added_sharding_axes=(batch_axis, None),
        )
        return DFlashDraftResult(hidden_states=hidden_states)


def make_qwen3_dflash_draft_config(
    *,
    model_dim: int,
    hidden_dim: int,
    block_size: int,
    mask_token_id: int,
    target_layer_ids: tuple[int, ...],
    num_target_layers: int,
    vocab_size: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    num_layers: int,
    rms_norm_eps: float,
    rope_config: RoPEConfig,
    layer_sliding_window_sizes: tuple[int | None, ...],
    attention_bias: bool,
    activation: Activation,
) -> DFlashDraftConfig:
    if len(layer_sliding_window_sizes) != num_layers:
        raise ValueError(
            f"Expected {num_layers} sliding-window entries, got {len(layer_sliding_window_sizes)}",
        )
    linear_config = LinearConfig()
    norm_config = NormalizationConfig(
        epsilon=rms_norm_eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    mlp_config = DenseMLPConfig(
        linear_config=linear_config,
        activation=activation,
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    layer_configs = tuple(
        DFlashDraftLayerConfig(
            attention_config=DFlashAttentionConfig(
                linear_config=linear_config,
                query_norm_config=norm_config,
                key_norm_config=norm_config,
                rope_config=rope_config,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                has_attention_biases=attention_bias,
                has_output_biases=attention_bias,
                sliding_window_size=sliding_window_size,
                scale=head_dim**-0.5,
            ),
            input_norm_config=norm_config,
            post_attention_norm_config=norm_config,
            mlp_config=mlp_config,
        )
        for sliding_window_size in layer_sliding_window_sizes
    )
    return DFlashDraftConfig(
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        block_size=block_size,
        mask_token_id=mask_token_id,
        target_layer_ids=target_layer_ids,
        num_target_layers=num_target_layers,
        vocab_size=vocab_size,
        context_projection_config=linear_config,
        context_norm_config=norm_config,
        layer_configs=layer_configs,
        output_norm_config=norm_config,
    )
