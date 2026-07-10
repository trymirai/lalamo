from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.weight_matrix import GradientEstimator

from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import Normalization, NormalizationConfig, NormalizationForwardPassConfig
from .ple import PLELayer, PLELayerConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
)
from .token_mixers.attention import Attention, AttentionConfig
from .utils import call_vmapped, call_vmapped_twice

__all__ = [
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
    outputs: Float[Array, "batch suffix_tokens channels"]
    updated_state: StateLayerBase | None
    activation_trace: TransformerLayerActivationTrace | None


@dataclass(frozen=True)
class TransformerLayerConfig(LalamoConfig):
    pre_mixer_norm_config: NormalizationConfig | None
    mixer_config: TokenMixerConfig
    post_mixer_norm_config: NormalizationConfig | None
    pre_mlp_norm_config: NormalizationConfig
    mlp_config: MLPConfig
    post_mlp_norm_config: NormalizationConfig | None
    hidden_dim: int | None = None
    parallel_mlp_config: MLPConfig | None = None
    mlp_output_norm_config: NormalizationConfig | None = None
    parallel_mlp_output_norm_config: NormalizationConfig | None = None
    ple_config: PLELayerConfig | None = None
    ple_norm_config: NormalizationConfig | None = None
    has_post_layer_scalar: bool = False
    rope_config: RoPEConfig | None = None

    def __post_init__(self) -> None:
        has_parallel_mlp = self.parallel_mlp_config is not None
        assert (self.mlp_output_norm_config is not None) == has_parallel_mlp
        assert (self.parallel_mlp_output_norm_config is not None) == has_parallel_mlp
        assert (self.ple_norm_config is not None) == (self.ple_config is not None)

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        hidden_dim: int,
        *,
        borrows_kv_cache: bool = False,
    ) -> "TransformerLayer":
        pre_mixer_norm = (
            self.pre_mixer_norm_config.init(initializer, model_dim) if self.pre_mixer_norm_config else None
        )
        if isinstance(self.mixer_config, AttentionConfig):
            mixer = self.mixer_config.init(
                initializer,
                model_dim=model_dim,
                borrows_kv_cache=borrows_kv_cache,
            )
        else:
            assert not borrows_kv_cache
            mixer = self.mixer_config.init(initializer, model_dim=model_dim)
        if self.rope_config is None:
            rope = None
        else:
            rope = self.rope_config.init(initializer)
        post_mixer_norm = (
            self.post_mixer_norm_config.init(initializer, model_dim) if self.post_mixer_norm_config else None
        )
        pre_mlp_norm = self.pre_mlp_norm_config.init(initializer, model_dim)
        mlp = self.mlp_config.init(initializer, model_dim, hidden_dim)
        parallel_mlp = (
            self.parallel_mlp_config.init(initializer, model_dim, hidden_dim) if self.parallel_mlp_config else None
        )
        mlp_output_norm = (
            self.mlp_output_norm_config.init(initializer, model_dim) if self.mlp_output_norm_config else None
        )
        parallel_mlp_output_norm = (
            self.parallel_mlp_output_norm_config.init(initializer, model_dim)
            if self.parallel_mlp_output_norm_config
            else None
        )
        post_mlp_norm = self.post_mlp_norm_config.init(initializer, model_dim) if self.post_mlp_norm_config else None
        ple = self.ple_config.init(initializer, model_dim) if self.ple_config else None
        ple_norm = self.ple_norm_config.init(initializer, model_dim) if self.ple_norm_config else None
        post_layer_scalar = initializer.ones((1,)) if self.has_post_layer_scalar else None
        return TransformerLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            pre_mixer_norm=pre_mixer_norm,
            mixer=mixer,
            rope=rope,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            parallel_mlp=parallel_mlp,
            mlp_output_norm=mlp_output_norm,
            parallel_mlp_output_norm=parallel_mlp_output_norm,
            post_mlp_norm=post_mlp_norm,
            ple=ple,
            ple_norm=ple_norm,
            post_layer_scalar=post_layer_scalar,
        )


class TransformerLayer(LalamoModule[TransformerLayerConfig]):
    pre_mixer_norm: Normalization | None
    mixer: TokenMixerBase
    rope: RoPE | None
    post_mixer_norm: Normalization | None
    pre_mlp_norm: Normalization
    mlp: MLPBase
    parallel_mlp: MLPBase | None
    mlp_output_norm: Normalization | None
    parallel_mlp_output_norm: Normalization | None
    post_mlp_norm: Normalization | None
    ple: PLELayer | None
    ple_norm: Normalization | None
    post_layer_scalar: Float[Array, "1"] | None

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
        per_layer_input: Float[Array, "batch suffix_tokens ple_channels"] | None = None,
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
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
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

        assert mlp_inputs.dtype == inputs.dtype

        normalized_mlp_inputs = call_vmapped_twice(
            self.pre_mlp_norm,
            mlp_inputs,
            forward_pass_config=normalization_forward_pass_config,
        )
        mlp_forward_pass_config = replace(
            forward_pass_config.mlp_forward_pass_config,
            normalization_forward_pass_config=normalization_forward_pass_config,
        )
        if self.parallel_mlp is None:
            primary_mlp_keychain = mlp_keychain
            parallel_mlp_keychain = None
        else:
            primary_mlp_keychain, parallel_mlp_keychain = mlp_keychain.split()

        mlp_outputs = self.mlp(
            normalized_mlp_inputs,
            lengths_without_padding=lengths_without_padding,
            forward_pass_config=mlp_forward_pass_config,
            keychain=primary_mlp_keychain,
        )
        if self.parallel_mlp is not None:
            assert parallel_mlp_keychain is not None
            assert self.mlp_output_norm is not None
            assert self.parallel_mlp_output_norm is not None
            mlp_outputs = call_vmapped_twice(
                self.mlp_output_norm,
                mlp_outputs,
                forward_pass_config=normalization_forward_pass_config,
            )
            parallel_mlp_outputs = self.parallel_mlp(
                normalized_mlp_inputs,
                lengths_without_padding=lengths_without_padding,
                forward_pass_config=mlp_forward_pass_config,
                keychain=parallel_mlp_keychain,
            )
            parallel_mlp_outputs = call_vmapped_twice(
                self.parallel_mlp_output_norm,
                parallel_mlp_outputs,
                forward_pass_config=normalization_forward_pass_config,
            )
            mlp_outputs = mlp_outputs + parallel_mlp_outputs

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
            assert self.ple_norm is not None
            ple_outputs = self.ple(
                outputs,
                per_layer_input,
                keychain=ple_keychain,
            )
            normalized_ple = call_vmapped_twice(
                self.ple_norm,
                ple_outputs,
                forward_pass_config=normalization_forward_pass_config,
            )
            outputs = outputs + normalized_ple

        if self.post_layer_scalar is not None:
            outputs = outputs * self.post_layer_scalar.astype(outputs.dtype)

        if return_activation_trace:
            activation_trace_state = state
            if isinstance(self.mixer, Attention) and self.mixer.borrows_kv_cache:
                activation_trace_state = None
            activation_trace = TransformerLayerActivationTrace(
                inputs=inputs,
                positional_embeddings=positional_embeddings,
                state=activation_trace_state,
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

        assert outputs.dtype == inputs.dtype
        return TransformerLayerResult(
            outputs=outputs,
            updated_state=updated_state,
            activation_trace=activation_trace,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> StateLayerBase:
        return jax.tree.map(
            lambda array: jax.device_put(
                jnp.repeat(array[None, ...], batch_size, axis=0),
                self.sharding_config.resolve_sharding((LogicalAxis.BATCH, *((None,) * array.ndim))),
            ),
            self.mixer.init_static_state(capacity, dtype),
        )
