from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, ShardingAxis
from lalamo.weight_matrix import GradientEstimator

from .activations import Activation
from .linear import Linear, LinearConfig
from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import Normalization, NormalizationConfig, NormalizationForwardPassConfig
from .rope import PositionalEmbeddings, RoPEConfig
from .token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
)
from .utils import call_vmapped, call_vmapped_twice

__all__ = [
    "PLELayer",
    "PLELayerConfig",
    "PositionalEmbeddingSelector",
    "TransformerForwardPassConfig",
    "TransformerLayer",
    "TransformerLayerActivationTrace",
    "TransformerLayerConfig",
    "TransformerLayerResult",
]


@dataclass(frozen=True)
class TransformerForwardPassConfig:
    mixer_forward_pass_config: MixerForwardPassConfig = dataclass_field(default_factory=MixerForwardPassConfig)
    mlp_forward_pass_config: MLPForwardPassConfig = dataclass_field(default_factory=MLPForwardPassConfig)
    normalization_forward_pass_config: NormalizationForwardPassConfig = dataclass_field(
        default_factory=NormalizationForwardPassConfig,
    )

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            mixer_forward_pass_config=MixerForwardPassConfig.for_tracer_tests(),
            mlp_forward_pass_config=MLPForwardPassConfig.for_tracer_tests(),
            normalization_forward_pass_config=NormalizationForwardPassConfig.for_tracer_tests(),
        )

    @classmethod
    def for_inference(
        cls,
        mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            mixer_forward_pass_config=MixerForwardPassConfig.for_inference(precision),
            mlp_forward_pass_config=MLPForwardPassConfig.for_inference(mode, precision),
            normalization_forward_pass_config=NormalizationForwardPassConfig.for_inference(),
        )

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            mixer_forward_pass_config=MixerForwardPassConfig.for_training(gradient_estimator, precision),
            mlp_forward_pass_config=MLPForwardPassConfig.for_training(gradient_estimator, precision),
            normalization_forward_pass_config=NormalizationForwardPassConfig.for_training(),
        )


class TransformerLayerActivationTrace(Exportable, eqx.Module):
    inputs: Float[Array, "batch suffix_tokens channels"]
    positional_embeddings: PositionalEmbeddings | None
    state: StateLayerBase | None

    mlp_inputs: Float[Array, "batch suffix_tokens channels"]
    pre_mixer_norm: Float[Array, "batch suffix_tokens channels"]
    mixer: Float[Array, "batch suffix_tokens channels"]
    post_mixer_norm: Float[Array, "batch suffix_tokens channels"] | None
    pre_mlp_norm: Float[Array, "batch suffix_tokens channels"]
    mlp: Float[Array, "batch suffix_tokens channels"]
    post_mlp_norm: Float[Array, "batch suffix_tokens channels"] | None


class TransformerLayerResult(Exportable, eqx.Module):
    outputs: Float[Array, "batch tokens channels"]
    updated_state: StateLayerBase | None
    activation_trace: TransformerLayerActivationTrace | None


@dataclass(frozen=True)
class PLELayerConfig(LalamoConfig):
    linear_config: LinearConfig
    norm_config: NormalizationConfig
    ple_dim: int
    activation: Activation

    def init(self, initializer: Initializer, model_dim: int) -> "PLELayer":
        gate = self.linear_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(self.ple_dim,),
            has_biases=False,
        )
        projection = self.linear_config.init(
            initializer,
            input_dim=self.ple_dim,
            output_dims=(model_dim,),
            has_biases=False,
        )
        norm = self.norm_config.init(initializer, model_dim)
        return PLELayer(config=self, gate=gate, projection=projection, norm=norm)


class PLELayer(LalamoModule[PLELayerConfig]):
    gate: Linear
    projection: Linear
    norm: Normalization

    def __call__(
        self,
        outputs: Float[Array, "batch suffix_tokens channels"],
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"],
        forward_pass_config: NormalizationForwardPassConfig = NormalizationForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        gate_keychain, projection_keychain = keychain.split()
        (ple_gated,) = call_vmapped_twice(
            self.gate,
            outputs,
            keychain=gate_keychain,
            added_sharding_axes=(ShardingAxis.DATA, None),
        )
        ple_gated = self.config.activation(ple_gated) * per_layer_input
        (ple_projected,) = call_vmapped_twice(
            self.projection,
            ple_gated,
            keychain=projection_keychain,
            added_sharding_axes=(ShardingAxis.DATA, None),
        )
        ple_normed = call_vmapped_twice(self.norm, ple_projected, forward_pass_config=forward_pass_config)
        return outputs + ple_normed


@dataclass(frozen=True)
class TransformerLayerConfig(LalamoConfig):
    pre_mixer_norm_config: NormalizationConfig | None
    mixer_config: TokenMixerConfig
    post_mixer_norm_config: NormalizationConfig | None
    pre_mlp_norm_config: NormalizationConfig
    mlp_config: MLPConfig
    post_mlp_norm_config: NormalizationConfig | None
    hidden_dim: int | None = None
    ple_config: PLELayerConfig | None = None
    has_post_layer_scalar: bool = False
    kv_source_layer_index: int | None = None
    rope_config: RoPEConfig | None = None

    @property
    def rope_dim(self) -> int | None:
        return self.mixer_config.rope_dim

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        hidden_dim: int,
    ) -> "TransformerLayer":
        pre_mixer_norm = (
            self.pre_mixer_norm_config.init(initializer, model_dim) if self.pre_mixer_norm_config else None
        )
        mixer = self.mixer_config.init(initializer, model_dim=model_dim)
        post_mixer_norm = (
            self.post_mixer_norm_config.init(initializer, model_dim) if self.post_mixer_norm_config else None
        )
        pre_mlp_norm = self.pre_mlp_norm_config.init(initializer, model_dim)
        mlp = self.mlp_config.init(initializer, model_dim, hidden_dim)
        post_mlp_norm = self.post_mlp_norm_config.init(initializer, model_dim) if self.post_mlp_norm_config else None
        ple = self.ple_config.init(initializer, model_dim) if self.ple_config else None
        post_layer_scalar = initializer.ones((1,)) if self.has_post_layer_scalar else None
        return TransformerLayer(
            config=self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=mixer,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
            ple=ple,
            post_layer_scalar=post_layer_scalar,
        )


class TransformerLayer(LalamoModule[TransformerLayerConfig]):
    pre_mixer_norm: Normalization | None
    mixer: TokenMixerBase
    post_mixer_norm: Normalization | None
    pre_mlp_norm: Normalization | None
    mlp: MLPBase
    post_mlp_norm: Normalization | None
    ple: PLELayer | None
    post_layer_scalar: Float[Array, "1"] | None

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return self.mixer.positional_embedding_selector

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: StateLayerBase | None = None,
        return_updated_state: bool = False,
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
        *,
        keychain: Keychain,
    ) -> TransformerLayerResult:
        if inputs.ndim != 3:
            raise ValueError(
                f"Inputs to decoder layers must be a 3D arrays of size (batch_size, sequence_length, hidden_dim),"
                f" got {inputs.shape}",
            )
        mixer_keychain, mlp_keychain, ple_keychain = keychain.split(3)
        normalization_forward_pass_config = forward_pass_config.normalization_forward_pass_config

        if self.pre_mixer_norm is not None:
            normalized_mixer_inputs = call_vmapped_twice(
                self.pre_mixer_norm,
                inputs,
                forward_pass_config=normalization_forward_pass_config,
            )
        else:
            normalized_mixer_inputs = inputs

        def call_mixer(
            mixer_inputs: tuple[
                Float[Array, "suffix_tokens channels"],
                PositionalEmbeddings | None,
                StateLayerBase | None,
                Int[Array, ""] | None,
                Int[Array, " suffix_tokens"] | None,
            ],
            *,
            keychain: Keychain,
        ) -> tuple[Float[Array, "suffix_tokens channels"], StateLayerBase | None]:
            mixer_input, positional_embedding, mixer_state, length_without_padding, parent_indices = mixer_inputs
            return self.mixer(
                mixer_input,
                positional_embedding,
                mixer_state,
                return_updated_state=return_updated_state or return_activation_trace,
                length_without_padding=length_without_padding,
                forward_pass_config=forward_pass_config.mixer_forward_pass_config,
                attention_parent_indices=parent_indices,
                keychain=keychain,
            )

        mixer_outputs, updated_state = call_vmapped(
            call_mixer,
            (
                normalized_mixer_inputs,
                positional_embeddings,
                state,
                lengths_without_padding,
                attention_parent_indices,
            ),
            keychain=mixer_keychain,
            added_sharding_axis=ShardingAxis.DATA,
        )
        if self.post_mixer_norm is not None:
            normalized_mixer_outputs = call_vmapped_twice(
                self.post_mixer_norm,
                mixer_outputs,
                forward_pass_config=normalization_forward_pass_config,
            )
            mlp_inputs = inputs + normalized_mixer_outputs
        else:
            normalized_mixer_outputs = None
            mlp_inputs = inputs + mixer_outputs

        normalized_mlp_inputs = (
            call_vmapped_twice(
                self.pre_mlp_norm,
                mlp_inputs,
                forward_pass_config=normalization_forward_pass_config,
            )
            if self.pre_mlp_norm is not None
            else mlp_inputs
        )
        mlp_outputs = self.mlp(
            normalized_mlp_inputs,
            lengths_without_padding=lengths_without_padding,
            forward_pass_config=forward_pass_config.mlp_forward_pass_config,
            keychain=mlp_keychain,
        )
        if self.post_mlp_norm is not None:
            normalized_mlp_outputs = call_vmapped_twice(
                self.post_mlp_norm,
                mlp_outputs,
                forward_pass_config=normalization_forward_pass_config,
            )
            outputs = mlp_inputs + normalized_mlp_outputs
        else:
            normalized_mlp_outputs = None
            outputs = mlp_inputs + mlp_outputs

        if self.ple is not None and per_layer_input is not None:
            outputs = self.ple(
                outputs,
                per_layer_input,
                forward_pass_config=normalization_forward_pass_config,
                keychain=ple_keychain,
            )
        if self.post_layer_scalar is not None:
            outputs = outputs * self.post_layer_scalar

        if return_activation_trace:
            activation_trace = TransformerLayerActivationTrace(
                inputs=inputs,
                positional_embeddings=positional_embeddings,
                state=state,
                pre_mixer_norm=normalized_mixer_inputs,
                mixer=mixer_outputs,
                post_mixer_norm=normalized_mixer_outputs,
                mlp_inputs=mlp_inputs,
                pre_mlp_norm=normalized_mlp_inputs,
                mlp=mlp_outputs,
                post_mlp_norm=normalized_mlp_outputs,
            )
        else:
            activation_trace = None

        return TransformerLayerResult(
            outputs=outputs,
            updated_state=updated_state,
            activation_trace=activation_trace,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> StateLayerBase:
        return jax.tree.map(
            lambda array: jnp.repeat(array[None, ...], batch_size, axis=0),
            self.mixer.init_static_state(capacity, dtype),
        )
