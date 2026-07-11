from dataclasses import dataclass
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, SpeculatorState
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.mlp import DenseMLP, DenseMLPConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings, RoPE, RoPEConfig
from lalamo.modules.speculator import Speculator, SpeculatorConfig
from lalamo.modules.speculators.weaver import Weaver, WeaverConfig
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from lalamo.modules.token_mixers.kv_cache import StaticKVCacheLayer
from lalamo.modules.transformer_layer import TransformerForwardPassConfig
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

__all__ = [
    "DFlashDraftConfig",
    "DFlashDraftLayer",
    "DFlashDraftLayerConfig",
    "DFlashDraftModel",
    "DFlashDraftState",
]


@dataclass(frozen=True)
class DFlashDraftLayerConfig(LalamoConfig):
    attention_config: AttentionConfig
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
    attention: Attention
    input_norm: Normalization
    post_attention_norm: Normalization
    mlp: DenseMLP

    def project_context(
        self,
        target_hidden: Float[Array, "batch context_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "batch context_tokens groups head_channels"],
        Float[Array, "batch context_tokens groups head_channels"],
    ]:
        return call_vmapped(
            self.attention.project_key_value_heads,
            target_hidden,
            positional_embeddings,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config,
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )

    @eqx.filter_jit
    def __call__(
        self,
        hidden_states: Float[Array, "batch query_tokens channels"],
        context_state: StaticKVCacheLayer,
        positional_embeddings: PositionalEmbeddings,
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
        attention_results = call_vmapped(
            self.attention,
            normalized_attention_inputs,
            positional_embeddings,
            context_state,
            forward_pass_config=forward_pass_config.mixer_forward_pass_config,
            keychain=attention_keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        mlp_inputs = hidden_states + attention_results.outputs
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
    rope_config: RoPEConfig
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
            rope=self.rope_config.init(initializer),
            layers=tuple(
                layer_config.init(initializer, self.model_dim, self.hidden_dim) for layer_config in self.layer_configs
            ),
            output_norm=self.output_norm_config.init(initializer, self.model_dim),
        )


class DFlashDraftState(SpeculatorState):
    layer_states: tuple[StaticKVCacheLayer, ...]

    @property
    def context_lengths(self) -> Int[Array, " batch"]:
        first_layer_state, *_ = self.layer_states
        return first_layer_state.current_length

    def append(
        self,
        layer_key_values: tuple[
            tuple[
                Float[Array, "batch tokens groups head_channels"],
                Float[Array, "batch tokens groups head_channels"],
            ],
            ...,
        ],
        num_tokens_to_append: Int[Array, " batch"],
        context_capacity: int,
        cache_sharding: NamedSharding,
    ) -> Self:
        context_lengths = self.context_lengths
        batch_size = context_lengths.shape[0]
        batch_indices = jnp.arange(batch_size, dtype=context_lengths.dtype)[:, None]
        updated_lengths = jnp.minimum(context_lengths + num_tokens_to_append, context_capacity)

        def scattered(
            buffer: Float[Array, "batch total_capacity groups head_channels"],
            update: Float[Array, "batch tokens groups head_channels"],
        ) -> Float[Array, "batch total_capacity groups head_channels"]:
            _, num_update_tokens, _, _ = update.shape
            update_offsets = jnp.arange(num_update_tokens, dtype=context_lengths.dtype)[None, :]
            destination_indices = context_lengths[:, None] + update_offsets
            is_valid = (update_offsets < num_tokens_to_append[:, None]) & (destination_indices < context_capacity)
            masked_update = jnp.where(is_valid[:, :, None, None], update.astype(buffer.dtype), 0)
            return buffer.at[batch_indices, destination_indices].set(
                masked_update,
                mode="drop",
                out_sharding=cache_sharding,
            )

        return DFlashDraftState(
            layer_states=tuple(
                StaticKVCacheLayer(
                    has_sinks=layer_state.has_sinks,
                    keys=scattered(layer_state.keys, added_keys),
                    values=scattered(layer_state.values, added_values),
                    current_length=updated_lengths,
                )
                for layer_state, (added_keys, added_values) in zip(
                    self.layer_states,
                    layer_key_values,
                    strict=True,
                )
            ),
        )


class DFlashDraftModel(LalamoModule[DFlashDraftConfig]):
    context_projection: Linear
    context_norm: Normalization
    rope: RoPE
    layers: tuple[DFlashDraftLayer, ...]
    output_norm: Normalization

    def positional_embeddings(
        self,
        token_positions: Int[Array, "batch tokens"],
        forward_pass_config: TransformerForwardPassConfig,
    ) -> PositionalEmbeddings:
        embeddings = call_vmapped(
            self.rope,
            token_positions,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        return embeddings.astype(forward_pass_config.mixer_forward_pass_config.rope_dtype)

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
        total_capacity = context_capacity + self.config.block_size

        def empty_layer_state(attention_config: AttentionConfig) -> StaticKVCacheLayer:
            cache = jax.device_put(
                jnp.zeros(
                    (
                        batch_size,
                        total_capacity,
                        attention_config.num_groups,
                        attention_config.head_dim,
                    ),
                    dtype=dtype,
                ),
                cache_sharding,
            )
            return StaticKVCacheLayer(
                has_sinks=False,
                keys=cache,
                values=cache,
                current_length=jax.device_put(jnp.zeros((batch_size,), dtype=jnp.int32), lengths_sharding),
            )

        return DFlashDraftState(
            layer_states=tuple(empty_layer_state(layer.attention.config) for layer in self.layers),
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
        positional_embeddings = self.positional_embeddings(token_positions, forward_pass_config)
        cache_sharding = self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None, None))
        first_layer_state, *_ = state.layer_states
        _, total_capacity, _, _ = first_layer_state.keys.shape
        return state.append(
            tuple(
                layer.project_context(
                    target_hidden,
                    positional_embeddings,
                    forward_pass_config=forward_pass_config,
                    keychain=layer_keychain,
                )
                for layer, layer_keychain in zip(self.layers, layer_keychains, strict=True)
            ),
            num_tokens_to_append,
            total_capacity - self.config.block_size,
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
        draft_positions = (
            last_token_indices[:, None] + jnp.arange(1, block_size + 1, dtype=last_token_indices.dtype)[None, :]
        )
        positional_embeddings = self.positional_embeddings(draft_positions, forward_pass_config)

        layer_keychains = keychain.split(len(self.layers))
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)

        hidden_states = noise_embeddings
        for layer, layer_state, layer_keychain in zip(self.layers, state.layer_states, layer_keychains, strict=True):
            hidden_states = layer(
                hidden_states,
                layer_state,
                positional_embeddings,
                forward_pass_config=forward_pass_config,
                keychain=layer_keychain,
            )

        return call_vmapped_twice(
            self.output_norm,
            hidden_states,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
            added_sharding_axes=(batch_axis, None),
        )


@dataclass(frozen=True)
class DFlashSpeculatorConfig(SpeculatorConfig):
    draft_config: DFlashDraftConfig
    weaver_config: WeaverConfig | None

    def __post_init__(self) -> None:
        if self.weaver_config is None:
            return
        if self.weaver_config.d_model != self.draft_config.model_dim:
            raise ValueError(
                f"Weaver d_model {self.weaver_config.d_model} does not match"
                f" draft model_dim {self.draft_config.model_dim}.",
            )
        if self.weaver_config.k > self.draft_config.block_size - 1:
            raise ValueError(
                f"Weaver depth k={self.weaver_config.k} exceeds the draft block's"
                f" {self.draft_config.block_size - 1} proposal positions.",
            )

    def init(self, initializer: Initializer) -> "DFlashSpeculator":
        return DFlashSpeculator(
            config=self,
            sharding_config=initializer.sharding_config,
            draft_model=self.draft_config.init(initializer),
            weaver=self.weaver_config.init(initializer) if self.weaver_config is not None else None,
        )


class DFlashSpeculator(Speculator[DFlashSpeculatorConfig]):
    draft_model: DFlashDraftModel
    weaver: Weaver | None
