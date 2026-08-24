from dataclasses import dataclass
from math import sqrt
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, SpeculatorState
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings, RoPE, RoPEConfig
from lalamo.modules.speculator import Speculator, SpeculatorConfig
from lalamo.modules.speculators.weaver import Weaver, WeaverConfig
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from lalamo.modules.token_mixers.kv_cache import StaticKVCacheLayer
from lalamo.modules.transformer_layer import TransformerForwardPassConfig, TransformerLayer, TransformerLayerConfig
from lalamo.modules.utils import call_vmapped, call_vmapped_twice
from lalamo.weight_matrix import MatmulConfig

__all__ = [
    "CandidateSelector",
    "CandidateSelectorConfig",
    "DFlashDraftConfig",
    "DFlashDraftModel",
    "DFlashDraftState",
    "DFlashGroupedConvolution",
    "DFlashGroupedConvolutionConfig",
    "DFlashLayerGroupedConvolutions",
]


def _layer_attention_config(layer_config: TransformerLayerConfig) -> AttentionConfig:
    mixer_config = layer_config.mixer_config
    if not isinstance(mixer_config, AttentionConfig):
        raise TypeError(f"DFlash draft layers must use attention mixers, got {type(mixer_config).__name__}.")
    return mixer_config


def _layer_attention(layer: TransformerLayer) -> Attention:
    mixer = layer.mixer
    if not isinstance(mixer, Attention):
        raise TypeError(f"DFlash draft layers must use attention mixers, got {type(mixer).__name__}.")
    return mixer


@dataclass(frozen=True)
class DFlashGroupedConvolutionConfig(LalamoConfig):
    kernel_size: int
    group_size: int
    kernel_projection_config: LinearConfig

    def init(self, initializer: Initializer, model_dim: int) -> "DFlashGroupedConvolution":
        if model_dim % self.group_size != 0:
            raise ValueError(f"group_size {self.group_size} must divide model_dim {model_dim}.")
        num_groups = model_dim // self.group_size
        return DFlashGroupedConvolution(
            config=self,
            sharding_config=initializer.sharding_config,
            base_kernel=initializer.normal(
                1.0 / sqrt(self.kernel_size),
                (2, self.kernel_size, model_dim),
            ),
            kernel_projection=self.kernel_projection_config.init(
                initializer,
                input_dim=model_dim,
                output_dims=(2 * self.kernel_size * num_groups,),
                has_biases=False,
                is_sharded=False,
            ),
        )


class DFlashGroupedConvolution(LalamoModule[DFlashGroupedConvolutionConfig]):
    base_kernel: Float[Array, "sides taps channels"]
    kernel_projection: Linear

    @property
    def num_groups(self) -> int:
        _, _, model_dim = self.base_kernel.shape
        return model_dim // self.config.group_size

    def convolve(
        self,
        hidden_states: Float[Array, "batch block channels"],
        coefficient_deltas: Float[Array, "batch block taps groups"],
        side: int,
    ) -> Float[Array, "batch block channels"]:
        batch_size, block_size, model_dim = hidden_states.shape
        hidden_groups = hidden_states.reshape(
            batch_size,
            block_size,
            self.num_groups,
            self.config.group_size,
        )
        base_kernel = self.base_kernel[side].reshape(
            self.config.kernel_size,
            self.num_groups,
            self.config.group_size,
        )
        coefficients = base_kernel.astype(hidden_states.dtype)[None, None] + coefficient_deltas[..., None]
        outputs = coefficients[:, :, 0] * hidden_groups
        for tap in range(1, self.config.kernel_size):
            shifted = jnp.pad(
                hidden_groups[:, :-tap],
                ((0, 0), (tap, 0), (0, 0), (0, 0)),
            )
            outputs = outputs + coefficients[:, :, tap] * shifted
        return outputs.reshape(batch_size, block_size, model_dim)

    def prepare(
        self,
        inputs: Float[Array, "batch block channels"],
        forward_pass_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch block channels"], Float[Array, "batch block taps groups"]]:
        batch_size, block_size, _ = inputs.shape
        (projected_coefficients,) = call_vmapped_twice(
            self.kernel_projection,
            inputs,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        coefficients = projected_coefficients.reshape(
            batch_size,
            block_size,
            2,
            self.config.kernel_size,
            self.num_groups,
        )
        return self.convolve(inputs, coefficients[:, :, 0], side=0), coefficients[:, :, 1]

    def finish(
        self,
        outputs: Float[Array, "batch block channels"],
        coefficients: Float[Array, "batch block taps groups"],
    ) -> Float[Array, "batch block channels"]:
        return self.convolve(outputs, coefficients, side=1)


class DFlashLayerGroupedConvolutions(eqx.Module):
    attention: DFlashGroupedConvolution
    mlp: DFlashGroupedConvolution


@dataclass(frozen=True)
class CandidateSelectorConfig(LalamoConfig):
    rank: int
    top_k: int
    hidden_projection_config: LinearConfig

    def init(self, initializer: Initializer, model_dim: int, vocab_size: int) -> "CandidateSelector":
        if self.top_k > vocab_size:
            raise ValueError(f"top_k {self.top_k} exceeds vocab_size {vocab_size}.")
        return CandidateSelector(
            config=self,
            sharding_config=initializer.sharding_config,
            predecessor_codebook=initializer.normal(1.0 / sqrt(self.rank), (vocab_size, self.rank)),
            successor_codebook=initializer.normal(1.0 / sqrt(self.rank), (vocab_size, self.rank)),
            hidden_projection=self.hidden_projection_config.init(
                initializer,
                input_dim=model_dim,
                output_dims=(self.rank,),
                has_biases=False,
                is_sharded=False,
            ),
        )


class CandidateSelector(LalamoModule[CandidateSelectorConfig]):
    predecessor_codebook: Float[Array, "vocabulary rank"]
    successor_codebook: Float[Array, "vocabulary rank"]
    hidden_projection: Linear


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
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig
    grouped_convolution_config: DFlashGroupedConvolutionConfig | None = None
    candidate_selector_config: CandidateSelectorConfig | None = None

    def __post_init__(self) -> None:
        if (
            self.grouped_convolution_config is not None
            and self.grouped_convolution_config.kernel_size > self.block_size
        ):
            raise ValueError(
                f"grouped convolution kernel_size {self.grouped_convolution_config.kernel_size}"
                f" exceeds block_size {self.block_size}.",
            )

    def init(self, initializer: Initializer) -> "DFlashDraftModel":
        context_feature_dim = len(self.target_layer_ids) * self.model_dim
        attention_configs = tuple(_layer_attention_config(layer_config) for layer_config in self.layer_configs)
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
            state_kv_projection=LinearConfig().init(
                initializer,
                self.model_dim,
                tuple(2 * config.num_groups * config.head_dim for config in attention_configs),
                has_biases=False,
            ),
            layers=tuple(
                layer_config.init(initializer, self.model_dim, self.hidden_dim) for layer_config in self.layer_configs
            ),
            output_norm=self.output_norm_config.init(initializer, self.model_dim),
            layer_grouped_convolutions=(
                tuple(
                    DFlashLayerGroupedConvolutions(
                        attention=self.grouped_convolution_config.init(initializer, self.model_dim),
                        mlp=self.grouped_convolution_config.init(initializer, self.model_dim),
                    )
                    for _ in self.layer_configs
                )
                if self.grouped_convolution_config is not None
                else None
            ),
            candidate_selector=(
                self.candidate_selector_config.init(initializer, self.model_dim, self.vocab_size)
                if self.candidate_selector_config is not None
                else None
            ),
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
    state_kv_projection: Linear
    layers: tuple[TransformerLayer, ...]
    output_norm: Normalization
    layer_grouped_convolutions: tuple[DFlashLayerGroupedConvolutions, ...] | None
    candidate_selector: CandidateSelector | None

    def state_kv_projection_from_layers(self, layers: tuple[TransformerLayer, ...]) -> Linear:
        qkvg_projections = tuple(_layer_attention(layer).qkvg_projection for layer in layers)
        key_value_weights = jnp.concatenate(
            tuple(projection.weights.decompress()[projection.output_dims[0] :] for projection in qkvg_projections),
            axis=0,
        )
        weights = qkvg_projections[0].weights.spec.compress(
            key_value_weights,
            key=jax.random.key(0),
            sharding_config=self.state_kv_projection.weights.sharding_config,
            is_sharded=self.state_kv_projection.weights.is_sharded,
        )
        return eqx.tree_at(lambda projection: projection.weights, self.state_kv_projection, weights)

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
            layer_states=tuple(empty_layer_state(_layer_attention(layer).config) for layer in self.layers),
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
                call_vmapped(
                    _layer_attention(layer).project_key_value_heads,
                    target_hidden,
                    positional_embeddings,
                    forward_pass_config=forward_pass_config.mixer_forward_pass_config,
                    keychain=layer_keychain,
                    added_sharding_axis=layer.sharding_config.resolve_axis(LogicalAxis.BATCH),
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
        for layer_index, (layer, layer_state, layer_keychain) in enumerate(
            zip(self.layers, state.layer_states, layer_keychains, strict=True),
        ):
            grouped_convolutions = (
                self.layer_grouped_convolutions[layer_index] if self.layer_grouped_convolutions is not None else None
            )
            layer_result = layer(
                hidden_states,
                positional_embeddings,
                layer_state,
                forward_pass_config=forward_pass_config,
                mixer_transform=grouped_convolutions.attention if grouped_convolutions is not None else None,
                mlp_transform=grouped_convolutions.mlp if grouped_convolutions is not None else None,
                keychain=layer_keychain,
            )
            hidden_states = layer_result.outputs

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
        if self.weaver_config.target_model_dim != self.draft_config.model_dim:
            raise ValueError(
                f"Weaver target_model_dim {self.weaver_config.target_model_dim} does not match"
                f" draft model_dim {self.draft_config.model_dim}.",
            )
        if self.weaver_config.max_depth > self.draft_config.block_size - 1:
            raise ValueError(
                f"Weaver max_depth {self.weaver_config.max_depth} exceeds the draft block's"
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
