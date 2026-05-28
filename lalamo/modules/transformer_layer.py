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
from lalamo.weight_matrix import GradientEstimator

from .activations import Activation
from .linear import Linear, LinearConfig
from .mlp import MixtureOfExperts, MixtureOfExpertsConfig, MLPBase, MLPConfig, MLPForwardPassConfig, SoftmaxRouting
from .normalization import Normalization, NormalizationConfig, NormalizationForwardPassConfig
from .rope import PositionalEmbeddings, RoPEConfig
from .token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
)
from .token_mixers.kv_cache import BorrowedKVCacheLayer
from .utils import call_vmapped, call_vmapped_twice

__all__ = [
    "Gemma4MoEBlock",
    "Gemma4MoEBlockConfig",
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
        forward_pass_config: NormalizationForwardPassConfig = NormalizationForwardPassConfig(),
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
        ple_normed = call_vmapped_twice(self.norm, ple_projected, forward_pass_config=forward_pass_config)
        return outputs + ple_normed


@dataclass(frozen=True)
class Gemma4MoEBlockConfig(LalamoConfig):
    moe_config: MixtureOfExpertsConfig
    norm_config: NormalizationConfig
    router_norm_epsilon: float

    def init(self, initializer: Initializer, model_dim: int) -> "Gemma4MoEBlock":
        moe = self.moe_config.init(initializer, model_dim, hidden_dim=0)
        return Gemma4MoEBlock(
            config=self,
            sharding_config=initializer.sharding_config,
            moe=moe,
            pre_moe_norm=self.norm_config.init(initializer, model_dim),
            post_dense_norm=self.norm_config.init(initializer, model_dim),
            post_moe_norm=self.norm_config.init(initializer, model_dim),
            router_scale=initializer.ones((model_dim,), dtype=moe.router.weights.dtype),
            per_expert_scale=initializer.ones((moe.config.num_routed_experts,), dtype=moe.router.weights.dtype),
        )


class Gemma4MoEBlock(LalamoModule[Gemma4MoEBlockConfig]):
    moe: MixtureOfExperts
    pre_moe_norm: Normalization
    post_dense_norm: Normalization
    post_moe_norm: Normalization
    router_scale: Float[Array, " channels"]
    per_expert_scale: Float[Array, " experts"]

    @property
    def model_dim(self) -> int:
        return self.moe.model_dim

    @property
    def num_experts(self) -> int:
        return self.moe.config.num_routed_experts

    @property
    def num_active_experts(self) -> int:
        return self.moe.config.num_active_routed_experts

    def __post_init__(self) -> None:
        if self.moe.router.weights.is_sharded:
            raise ValueError("Gemma 4 router must not shard the expert axis before top_k.")
        if self.moe.router.has_biases:
            raise ValueError("Gemma 4 router does not use biases.")
        if not isinstance(self.moe.config.routing_function, SoftmaxRouting):
            raise TypeError("Gemma 4 MoE uses softmax routing.")
        if self.moe.config.num_shared_experts != 0 or self.moe.gate is not None:
            raise ValueError("Gemma 4 MoE uses only routed experts.")
        if self.router_scale.shape != (self.model_dim,):
            raise ValueError(f"router_scale must have shape {(self.model_dim,)}, got {self.router_scale.shape}.")
        if self.per_expert_scale.shape != (self.num_experts,):
            raise ValueError(
                f"per_expert_scale must have shape {(self.num_experts,)}, got {self.per_expert_scale.shape}.",
            )

    def _active_experts(
        self,
        router_input: Float[Array, " channels"],
        forward_pass_config: MLPForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, " active_experts"], Int[Array, " active_experts"]]:
        router_input_dtype = router_input.dtype
        router_input = router_input.astype(jnp.float32)
        router_input = router_input * jax.lax.rsqrt(
            jnp.mean(jnp.square(router_input)) + self.config.router_norm_epsilon,
        )
        router_input = router_input.astype(router_input_dtype)
        router_input = router_input * self.router_scale.astype(router_input.dtype) * (self.model_dim**-0.5)
        (router_logits,) = self.moe.router(
            router_input,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        routing = self.moe.config.routing_function.call_unbatched(router_logits, num_active=self.num_active_experts)
        active_indices = jnp.flatnonzero(routing.expert_mask, size=self.num_active_experts)
        active_weights = routing.expert_weights[active_indices]
        expert_scales = self.per_expert_scale.astype(active_weights.dtype)
        return active_weights * expert_scales[active_indices], active_indices

    def __call__(
        self,
        residual_inputs: Float[Array, "batch suffix_tokens channels"],
        dense_outputs: Float[Array, "batch suffix_tokens channels"],
        forward_pass_config: TransformerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        normalization_forward_pass_config = forward_pass_config.normalization_forward_pass_config
        dense_outputs = call_vmapped_twice(
            self.post_dense_norm,
            dense_outputs,
            forward_pass_config=normalization_forward_pass_config,
        )
        expert_inputs = call_vmapped_twice(
            self.pre_moe_norm,
            residual_inputs,
            forward_pass_config=normalization_forward_pass_config,
        )

        flattened_router_inputs = rearrange(
            residual_inputs,
            "batch suffix_tokens channels -> (batch suffix_tokens) channels",
        )
        flattened_expert_inputs = rearrange(
            expert_inputs,
            "batch suffix_tokens channels -> (batch suffix_tokens) channels",
        )

        def apply_one_token(
            router_input: Float[Array, " channels"],
            expert_input: Float[Array, " channels"],
            *,
            keychain: Keychain,
        ) -> Float[Array, " channels"]:
            router_keychain, expert_keychain = keychain.split()
            active_weights, active_indices = self._active_experts(
                router_input,
                forward_pass_config.mlp_forward_pass_config,
                keychain=router_keychain,
            )
            return self.moe.experts.call_weighted_mixture_unbatched(
                expert_input,
                active_indices,
                active_weights,
                forward_pass_config.mlp_forward_pass_config,
                keychain=expert_keychain,
            )

        flattened_outputs = call_vmapped(
            apply_one_token,
            flattened_router_inputs,
            flattened_expert_inputs,
            keychain=keychain,
        )
        moe_outputs = rearrange(
            flattened_outputs,
            "(batch suffix_tokens) channels -> batch suffix_tokens channels",
            batch=residual_inputs.shape[0],
        )
        moe_outputs = call_vmapped_twice(
            self.post_moe_norm,
            moe_outputs,
            forward_pass_config=normalization_forward_pass_config,
        )
        return dense_outputs + moe_outputs


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
    gemma4_moe_config: Gemma4MoEBlockConfig | None = None
    has_post_layer_scalar: bool = False
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
        gemma4_moe = self.gemma4_moe_config.init(initializer, model_dim) if self.gemma4_moe_config else None
        post_layer_scalar = initializer.ones((1,)) if self.has_post_layer_scalar else None
        return TransformerLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            pre_mixer_norm=pre_mixer_norm,
            mixer=mixer,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
            ple=ple,
            gemma4_moe=gemma4_moe,
            post_layer_scalar=post_layer_scalar,
        )


class TransformerLayer(LalamoModule[TransformerLayerConfig]):
    pre_mixer_norm: Normalization | None
    mixer: TokenMixerBase
    post_mixer_norm: Normalization | None
    pre_mlp_norm: Normalization
    mlp: MLPBase
    post_mlp_norm: Normalization | None
    ple: PLELayer | None
    gemma4_moe: Gemma4MoEBlock | None
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
        *,
        keychain: Keychain,
    ) -> TransformerLayerResult:
        if inputs.ndim != 3:
            raise ValueError(
                f"Inputs to decoder layers must be a 3D arrays of size (batch_size, sequence_length, hidden_dim),"
                f" got {inputs.shape}",
            )
        mixer_keychain, mlp_keychain, moe_keychain, ple_keychain = keychain.split(4)
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
                reuse_cache=isinstance(state, BorrowedKVCacheLayer),
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
        mlp_outputs = self.mlp(
            normalized_mlp_inputs,
            lengths_without_padding=lengths_without_padding,
            forward_pass_config=forward_pass_config.mlp_forward_pass_config,
            keychain=mlp_keychain,
        )
        if self.gemma4_moe is not None:
            mlp_outputs = self.gemma4_moe(
                mlp_inputs,
                mlp_outputs,
                forward_pass_config=forward_pass_config,
                keychain=moe_keychain,
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
            outputs = outputs * self.post_layer_scalar.astype(outputs.dtype)

        if return_activation_trace:
            activation_trace = TransformerLayerActivationTrace(
                inputs=inputs,
                positional_embeddings=positional_embeddings,
                state=None if isinstance(state, BorrowedKVCacheLayer) else state,
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
