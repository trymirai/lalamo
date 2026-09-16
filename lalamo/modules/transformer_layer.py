from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.weight_matrix import GradientEstimator, MatmulConfig

from .activations import Activation
from .linear import Linear, LinearConfig
from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPEConfig
from .token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
)
from .token_mixers.convolutions import SeparableCausalConv, SeparableCausalConvConfig
from .utils import call_vmapped, call_vmapped_twice, gather_suffix_tokens

__all__ = [
    "PLELayer",
    "PLELayerConfig",
    "PositionalEmbeddingSelector",
    "TransformerForwardPassConfig",
    "TransformerLayer",
    "TransformerLayerActivationTrace",
    "TransformerLayerConfig",
    "TransformerLayerConv",
    "TransformerLayerConvConfig",
    "TransformerLayerResult",
]


@dataclass(frozen=True)
class TransformerForwardPassConfig:
    mixer_forward_pass_config: MixerForwardPassConfig = dataclass_field(default_factory=MixerForwardPassConfig)
    mlp_forward_pass_config: MLPForwardPassConfig = dataclass_field(default_factory=MLPForwardPassConfig)

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            mixer_forward_pass_config=MixerForwardPassConfig.for_tracer_tests(),
            mlp_forward_pass_config=MLPForwardPassConfig.for_tracer_tests(),
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
        return PLELayer(
            config=self,
            sharding_config=initializer.sharding_config,
            gate=gate,
            projection=projection,
            norm=norm,
        )


class PLELayer(LalamoModule[PLELayerConfig]):
    gate: Linear
    projection: Linear
    norm: Normalization

    def __call__(
        self,
        outputs: Float[Array, "batch suffix_tokens channels"],
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        gate_keychain, projection_keychain = keychain.split()
        (ple_gated,) = call_vmapped_twice(
            self.gate,
            outputs,
            keychain=gate_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        ple_gated = self.config.activation(ple_gated) * per_layer_input
        (ple_projected,) = call_vmapped_twice(
            self.projection,
            ple_gated,
            keychain=projection_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        ple_normed = call_vmapped_twice(self.norm, ple_projected)
        return outputs + ple_normed


@dataclass(frozen=True)
class TransformerLayerConvConfig(LalamoConfig):
    conv_config: SeparableCausalConvConfig
    kernel_projection_config: LinearConfig
    conv_kernel_size: int
    conv_group_size: int

    def init(self, initializer: Initializer, model_dim: int) -> "TransformerLayerConv":
        if model_dim % self.conv_group_size != 0:
            raise ValueError(f"conv_group_size {self.conv_group_size} must divide model_dim {model_dim}.")
        pre_conv = self.conv_config.init(
            initializer, model_dim, self.conv_kernel_size, dtype=initializer.default_dtype
        )
        post_conv = self.conv_config.init(
            initializer, model_dim, self.conv_kernel_size, dtype=initializer.default_dtype
        )
        kernel_projection = self.kernel_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(2 * self.conv_kernel_size * (model_dim // self.conv_group_size),),
            has_biases=False,
            is_sharded=False,
        )
        return TransformerLayerConv(
            config=self,
            sharding_config=initializer.sharding_config,
            kernel_projection=kernel_projection,
            pre_conv=pre_conv,
            post_conv=post_conv,
        )


class TransformerLayerConv(LalamoModule[TransformerLayerConvConfig]):
    kernel_projection: Linear
    pre_conv: SeparableCausalConv
    post_conv: SeparableCausalConv

    def prepare(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        forward_pass_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens channels"], Float[Array, "batch suffix_tokens kernel groups"]]:
        (projected_coefficients,) = call_vmapped_twice(
            self.kernel_projection,
            inputs,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        coefficients = jnp.flip(
            rearrange(
                projected_coefficients,
                "batch tokens (sides kernel groups) -> batch tokens sides kernel groups",
                sides=2,
                kernel=self.pre_conv.kernel_size,
            ),
            axis=3,
        )
        outputs = call_vmapped(
            lambda inputs, deltas: self.pre_conv(inputs, coefficient_deltas=deltas).outputs,
            inputs,
            coefficients[:, :, 0],
        )
        return outputs, coefficients[:, :, 1]

    def finish(
        self,
        outputs: Float[Array, "batch suffix_tokens channels"],
        coefficients: Float[Array, "batch suffix_tokens kernel groups"],
    ) -> Float[Array, "batch suffix_tokens channels"]:
        return call_vmapped(
            lambda outputs, deltas: self.post_conv(outputs, coefficient_deltas=deltas).outputs,
            outputs,
            coefficients,
        )


@dataclass(frozen=True)
class TransformerLayerConfig(LalamoConfig):
    pre_mixer_norm_config: NormalizationConfig | None
    mixer_conv_config: TransformerLayerConvConfig | None = dataclass_field(default=None, kw_only=True)
    mixer_config: TokenMixerConfig
    post_mixer_norm_config: NormalizationConfig | None
    pre_mlp_norm_config: NormalizationConfig
    mlp_conv_config: TransformerLayerConvConfig | None = dataclass_field(default=None, kw_only=True)
    mlp_config: MLPConfig
    post_mlp_norm_config: NormalizationConfig | None
    hidden_dim: int | None = None
    ple_config: PLELayerConfig | None = None
    has_post_layer_scalar: bool = False
    kv_source_layer_index: int | None = None
    rope_config: RoPEConfig | None = None

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
        mixer_conv = self.mixer_conv_config.init(initializer, model_dim) if self.mixer_conv_config else None
        mlp_conv = self.mlp_conv_config.init(initializer, model_dim) if self.mlp_conv_config else None
        return TransformerLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            pre_mixer_norm=pre_mixer_norm,
            mixer_conv=mixer_conv,
            mixer=mixer,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp_conv=mlp_conv,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
            ple=ple,
            post_layer_scalar=post_layer_scalar,
        )


class TransformerLayer(LalamoModule[TransformerLayerConfig]):
    pre_mixer_norm: Normalization | None
    mixer_conv: TransformerLayerConv | None
    mixer: TokenMixerBase
    post_mixer_norm: Normalization | None
    pre_mlp_norm: Normalization
    mlp_conv: TransformerLayerConv | None
    mlp: MLPBase
    post_mlp_norm: Normalization | None
    ple: PLELayer | None
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
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
        return_suffix_tokens: int | None = None,
        *,
        keychain: Keychain,
    ) -> TransformerLayerResult:
        if inputs.ndim != 3:
            raise ValueError(
                f"Inputs to decoder layers must be a 3D arrays of size (batch_size, sequence_length, hidden_dim),"
                f" got {inputs.shape}",
            )
        if return_suffix_tokens is not None and return_activation_trace:
            raise ValueError("return_suffix_tokens cannot be combined with return_activation_trace.")
        mixer_keychain, mlp_keychain, ple_keychain = keychain.split(3)

        if self.pre_mixer_norm is not None:
            normalized_mixer_inputs = call_vmapped_twice(self.pre_mixer_norm, inputs)
        else:
            normalized_mixer_inputs = inputs

        if self.mixer_conv is not None:
            mixer_transform_keychain, mixer_keychain = mixer_keychain.split()
            transformed_mixer_inputs, mixer_transform_state = self.mixer_conv.prepare(
                normalized_mixer_inputs,
                forward_pass_config.mixer_forward_pass_config.matmul_config,
                keychain=mixer_transform_keychain,
            )
        else:
            transformed_mixer_inputs = normalized_mixer_inputs
            mixer_transform_state = None

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
                reuse_cache=self.config.kv_source_layer_index is not None,
                keychain=keychain,
            )

        mixer_outputs, updated_state = call_vmapped(
            call_mixer,
            (
                transformed_mixer_inputs,
                positional_embeddings,
                state,
                lengths_without_padding,
                attention_parent_indices,
            ),
            keychain=mixer_keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        if self.mixer_conv is not None:
            assert mixer_transform_state is not None
            mixer_outputs = self.mixer_conv.finish(mixer_outputs, mixer_transform_state)
        if self.post_mixer_norm is not None:
            normalized_mixer_outputs = call_vmapped_twice(self.post_mixer_norm, mixer_outputs)
            mlp_inputs = inputs + normalized_mixer_outputs
        else:
            normalized_mixer_outputs = None
            mlp_inputs = inputs + mixer_outputs

        assert mlp_inputs.dtype == inputs.dtype

        if return_suffix_tokens is not None and self.mlp_conv is None:
            mlp_inputs = gather_suffix_tokens(
                mlp_inputs,
                lengths_without_padding,
                return_suffix_tokens,
                self.sharding_config,
            )
            if per_layer_input is not None:
                per_layer_input = gather_suffix_tokens(
                    per_layer_input,
                    lengths_without_padding,
                    return_suffix_tokens,
                    self.sharding_config,
                )
            # The gathered window is tail-aligned, while MLP padding masks assume the valid tokens
            # form a prefix, so the window is treated as fully valid instead.
            mlp_lengths_without_padding = None
        else:
            mlp_lengths_without_padding = lengths_without_padding

        normalized_mlp_inputs = call_vmapped_twice(self.pre_mlp_norm, mlp_inputs)
        if self.mlp_conv is not None:
            mlp_transform_keychain, mlp_keychain = mlp_keychain.split()
            transformed_mlp_inputs, mlp_transform_state = self.mlp_conv.prepare(
                normalized_mlp_inputs,
                forward_pass_config.mlp_forward_pass_config.matmul_config,
                keychain=mlp_transform_keychain,
            )
        else:
            transformed_mlp_inputs = normalized_mlp_inputs
            mlp_transform_state = None
        mlp_outputs = self.mlp(
            transformed_mlp_inputs,
            lengths_without_padding=mlp_lengths_without_padding,
            forward_pass_config=forward_pass_config.mlp_forward_pass_config,
            keychain=mlp_keychain,
        )
        if self.mlp_conv is not None:
            assert mlp_transform_state is not None
            mlp_outputs = self.mlp_conv.finish(mlp_outputs, mlp_transform_state)
        if self.post_mlp_norm is not None:
            normalized_mlp_outputs = call_vmapped_twice(self.post_mlp_norm, mlp_outputs)
            outputs = mlp_inputs + normalized_mlp_outputs
        else:
            normalized_mlp_outputs = None
            outputs = mlp_inputs + mlp_outputs

        if return_suffix_tokens is not None and self.mlp_conv is not None:
            outputs = gather_suffix_tokens(
                outputs,
                lengths_without_padding,
                return_suffix_tokens,
                self.sharding_config,
            )
            if per_layer_input is not None:
                per_layer_input = gather_suffix_tokens(
                    per_layer_input,
                    lengths_without_padding,
                    return_suffix_tokens,
                    self.sharding_config,
                )

        if self.ple is not None and per_layer_input is not None:
            outputs = self.ple(
                outputs,
                per_layer_input,
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
