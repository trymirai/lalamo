from dataclasses import dataclass
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, SpeculatorState
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.mlp import DenseMLP, DenseMLPConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
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
    "DFlashDraftState",
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


class DFlashDraftLayerState(eqx.Module):
    keys: Float[Array, "batch context_capacity groups head_channels"]
    values: Float[Array, "batch context_capacity groups head_channels"]

    def append(
        self,
        updates: Self,
        context_lengths: Int[Array, " batch"],
        num_tokens_to_append: Int[Array, " batch"],
        cache_sharding: jax.sharding.Sharding,
    ) -> Self:
        batch_size, num_update_tokens, _num_groups, _head_dim = updates.keys.shape
        batch_indices = jnp.arange(batch_size, dtype=context_lengths.dtype)[:, None]
        update_offsets = jnp.arange(num_update_tokens, dtype=context_lengths.dtype)[None, :]
        destination_indices = context_lengths[:, None] + update_offsets
        is_valid = update_offsets < num_tokens_to_append[:, None]

        def scattered(
            buffer: Float[Array, "batch context_capacity groups head_channels"],
            update: Float[Array, "batch update_tokens groups head_channels"],
        ) -> Float[Array, "batch context_capacity groups head_channels"]:
            masked_update = jnp.where(is_valid[:, :, None, None], update.astype(buffer.dtype), 0)
            return buffer.at[batch_indices, destination_indices].set(
                masked_update,
                mode="drop",
                out_sharding=cache_sharding,
            )

        return DFlashDraftLayerState(
            keys=scattered(self.keys, updates.keys),
            values=scattered(self.values, updates.values),
        )


class DFlashAttention(LalamoModule[DFlashAttentionConfig]):
    query_projection: Linear
    key_value_projection: Linear
    output_projection: Linear
    query_norm: Normalization
    key_norm: Normalization
    rope: RoPE

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
        token_positions: Int[Array, " key_tokens"],
        num_query_tokens: int,
    ) -> Bool[Array, "query_tokens key_tokens"] | None:
        if self.config.sliding_window_size is None:
            return attention_mask
        query_positions = token_positions[-num_query_tokens:]
        window_radius = self.config.sliding_window_size // 2
        sliding_window_mask = (token_positions[None, :] >= query_positions[:, None] - window_radius) & (
            token_positions[None, :] <= query_positions[:, None] + window_radius
        )
        if attention_mask is None:
            return sliding_window_mask
        return attention_mask & sliding_window_mask

    def _project_key_values(
        self,
        source_states: Float[Array, "tokens channels"],
        forward_pass_config: TransformerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "tokens groups head_channels"],
        Float[Array, "tokens groups head_channels"],
    ]:
        key_projection, value_projection = call_vmapped(
            self.key_value_projection,
            source_states,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config.matmul_config,
            keychain=keychain,
        )
        keys = self._reshape_keys_or_values(key_projection)
        values = self._reshape_keys_or_values(value_projection)
        return self._normalize_heads(self.key_norm, keys, forward_pass_config), values

    @eqx.filter_jit
    def project_context_unbatched(
        self,
        target_hidden: Float[Array, "context_tokens channels"],
        token_positions: Int[Array, " context_tokens"],
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "context_tokens groups head_channels"],
        Float[Array, "context_tokens groups head_channels"],
    ]:
        _query_keychain, key_value_keychain, _output_keychain = keychain.split(3)
        keys, values = self._project_key_values(
            target_hidden,
            forward_pass_config,
            keychain=key_value_keychain,
        )
        positional_embeddings = self.rope(token_positions).astype(
            forward_pass_config.mixer_forward_pass_config.rope_dtype,
        )
        return self._apply_rope(keys, positional_embeddings), values

    @eqx.filter_jit
    def project_context(
        self,
        target_hidden: Float[Array, "batch context_tokens channels"],
        token_positions: Int[Array, "batch context_tokens"],
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> DFlashDraftLayerState:
        keys, values = call_vmapped(
            self.project_context_unbatched,
            target_hidden,
            token_positions,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        return DFlashDraftLayerState(keys=keys, values=values)

    @eqx.filter_jit
    def call_unbatched(
        self,
        hidden_states: Float[Array, "query_tokens channels"],
        context_keys: Float[Array, "context_capacity groups head_channels"],
        context_values: Float[Array, "context_capacity groups head_channels"],
        token_positions: Int[Array, " key_tokens"],
        attention_mask: Bool[Array, "query_tokens key_tokens"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "query_tokens channels"]:
        query_keychain, key_value_keychain, output_keychain = keychain.split(3)
        num_query_tokens, _ = hidden_states.shape
        noise_token_positions = token_positions[-num_query_tokens:]

        (query_projection,) = call_vmapped(
            self.query_projection,
            hidden_states,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config.matmul_config,
            keychain=query_keychain,
        )
        noise_keys, noise_values = self._project_key_values(
            hidden_states,
            forward_pass_config,
            keychain=key_value_keychain,
        )

        queries = rearrange(
            query_projection,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        queries = self._normalize_heads(self.query_norm, queries, forward_pass_config)

        noise_positional_embeddings = self.rope(noise_token_positions).astype(
            forward_pass_config.mixer_forward_pass_config.rope_dtype,
        )
        queries = self._apply_rope(queries, noise_positional_embeddings)
        noise_keys = self._apply_rope(noise_keys, noise_positional_embeddings)
        keys = jnp.concatenate((context_keys, noise_keys.astype(context_keys.dtype)), axis=0)
        values = jnp.concatenate((context_values, noise_values.astype(context_values.dtype)), axis=0)

        effective_mask = self._attention_mask(attention_mask, token_positions, num_query_tokens)
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
        ).astype(hidden_states.dtype)
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
        context_state: DFlashDraftLayerState,
        token_positions: Int[Array, "batch key_tokens"],
        attention_mask: Bool[Array, "batch query_tokens key_tokens"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch query_tokens channels"]:
        return call_vmapped(
            self.call_unbatched,
            hidden_states,
            context_state.keys,
            context_state.values,
            token_positions,
            attention_mask,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
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

    def project_context(
        self,
        target_hidden: Float[Array, "batch context_tokens channels"],
        token_positions: Int[Array, "batch context_tokens"],
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> DFlashDraftLayerState:
        attention_keychain, _mlp_keychain = keychain.split(2)
        return self.attention.project_context(
            target_hidden,
            token_positions,
            forward_pass_config=forward_pass_config,
            keychain=attention_keychain,
        )

    @eqx.filter_jit
    def __call__(
        self,
        hidden_states: Float[Array, "batch query_tokens channels"],
        context_state: DFlashDraftLayerState,
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
            context_state,
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


class DFlashDraftState(SpeculatorState):
    layer_states: tuple[DFlashDraftLayerState, ...]
    context_lengths: Int[Array, " batch"]

    def append(
        self,
        layer_updates: tuple[DFlashDraftLayerState, ...],
        num_tokens_to_append: Int[Array, " batch"],
        cache_sharding: jax.sharding.Sharding,
    ) -> Self:
        return DFlashDraftState(
            layer_states=tuple(
                layer_state.append(
                    layer_update,
                    self.context_lengths,
                    num_tokens_to_append,
                    cache_sharding,
                )
                for layer_state, layer_update in zip(self.layer_states, layer_updates, strict=True)
            ),
            context_lengths=self.context_lengths + num_tokens_to_append,
        )


class DFlashDraftModel(LalamoModule[DFlashDraftConfig]):
    context_projection: Linear
    context_norm: Normalization
    layers: tuple[DFlashDraftLayer, ...]
    output_norm: Normalization

    @eqx.filter_jit
    def project_target_features(
        self,
        target_features: Float[Array, "batch tokens target_channels"],
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch tokens channels"]:
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        (target_hidden,) = call_vmapped_twice(
            self.context_projection,
            target_features,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config.matmul_config,
            keychain=keychain,
            added_sharding_axes=(batch_axis, None),
        )
        return call_vmapped_twice(
            self.context_norm,
            target_hidden,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
            added_sharding_axes=(batch_axis, None),
        )

    def empty_state(
        self,
        batch_size: int,
        context_capacity: int,
        dtype: DTypeLike,
    ) -> DFlashDraftState:
        cache_sharding = self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None, None))
        lengths_sharding = self.sharding_config.resolve_sharding((LogicalAxis.BATCH,))

        def empty_layer_state(attention_config: DFlashAttentionConfig) -> DFlashDraftLayerState:
            cache = jax.device_put(
                jnp.zeros(
                    (
                        batch_size,
                        context_capacity,
                        attention_config.num_key_value_heads,
                        attention_config.head_dim,
                    ),
                    dtype=dtype,
                ),
                cache_sharding,
            )
            return DFlashDraftLayerState(keys=cache, values=cache)

        return DFlashDraftState(
            layer_states=tuple(empty_layer_state(layer.attention.config) for layer in self.layers),
            context_lengths=jax.device_put(jnp.zeros((batch_size,), dtype=jnp.int32), lengths_sharding),
        )

    @eqx.filter_jit
    def append_state(
        self,
        state: DFlashDraftState,
        target_features: Float[Array, "batch tokens target_channels"],
        token_positions: Int[Array, "batch tokens"],
        num_tokens_to_append: Int[Array, " batch"],
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> DFlashDraftState:
        context_keychain, *layer_keychains = keychain.split(len(self.layers) + 1)
        target_hidden = self.project_target_features(
            target_features,
            forward_pass_config,
            keychain=context_keychain,
        )
        cache_sharding = self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None, None))
        return state.append(
            tuple(
                layer.project_context(
                    target_hidden,
                    token_positions,
                    forward_pass_config=forward_pass_config,
                    keychain=layer_keychain,
                )
                for layer, layer_keychain in zip(self.layers, layer_keychains, strict=True)
            ),
            num_tokens_to_append,
            cache_sharding,
        )

    @eqx.filter_jit
    def __call__(
        self,
        noise_embeddings: Float[Array, "batch block channels"],
        state: DFlashDraftState,
        last_token_indices: Int[Array, " batch"],
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch block channels"]:
        block_size = self.config.block_size
        batch_size, _, _ = noise_embeddings.shape
        first_layer_state, *_ = state.layer_states
        _, context_capacity, _, _ = first_layer_state.keys.shape
        context_slots = jnp.arange(context_capacity, dtype=state.context_lengths.dtype)[None, :]
        context_positions = (
            last_token_indices[:, None]
            - state.context_lengths[:, None]
            + 1
            + context_slots.astype(last_token_indices.dtype)
        )
        draft_positions = (
            last_token_indices[:, None] + jnp.arange(1, block_size + 1, dtype=last_token_indices.dtype)[None, :]
        )
        token_positions = jnp.concatenate((context_positions, draft_positions), axis=1)
        draft_key_mask = jnp.broadcast_to(
            jnp.ones_like(state.context_lengths[:, None], dtype=bool),
            (batch_size, block_size),
        )
        key_mask = jnp.concatenate(
            (context_slots < state.context_lengths[:, None], draft_key_mask),
            axis=1,
        )
        attention_mask = jnp.broadcast_to(
            key_mask[:, None, :], (batch_size, block_size, context_capacity + block_size)
        )

        layer_keychains = keychain.split(len(self.layers))
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)

        hidden_states = noise_embeddings
        for layer, layer_state, layer_keychain in zip(self.layers, state.layer_states, layer_keychains, strict=True):
            hidden_states = layer(
                hidden_states,
                layer_state,
                token_positions,
                attention_mask,
                forward_pass_config=forward_pass_config,
                keychain=layer_keychain,
            )

        return call_vmapped_twice(
            self.output_norm,
            hidden_states,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
            added_sharding_axes=(batch_axis, None),
        )
