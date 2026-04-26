from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule

from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings
from .token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
)
from .utils import call_vmapped, call_vmapped_twice

__all__ = [
    "PositionalEmbeddingSelector",
    "TransformerLayer",
    "TransformerLayerActivationTrace",
    "TransformerLayerConfig",
    "TransformerLayerForwardPassConfig",
    "TransformerLayerResult",
]


@dataclass(frozen=True)
class TransformerLayerForwardPassConfig:
    mixer: MixerForwardPassConfig = dataclass_field(default_factory=MixerForwardPassConfig)
    mlp: MLPForwardPassConfig = dataclass_field(default_factory=MLPForwardPassConfig)


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
    kv_source_layer: int | None = None
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
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: TransformerLayerForwardPassConfig = TransformerLayerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> TransformerLayerResult:
        if inputs.ndim != 3:
            raise ValueError(
                f"Inputs to decoder layers must be a 3D arrays of size (batch_size, sequence_length, hidden_dim),"
                f" got {inputs.shape}",
            )
        mixer_keychain, mlp_keychain = keychain.split()

        if self.pre_mixer_norm is not None:
            normalized_mixer_inputs = call_vmapped_twice(self.pre_mixer_norm, inputs)
        else:
            normalized_mixer_inputs = inputs

        def call_mixer(
            mixer_inputs: tuple[
                Float[Array, "suffix_tokens channels"],
                PositionalEmbeddings | None,
                StateLayerBase | None,
                Int[Array, ""] | None,
            ],
        ) -> tuple[Float[Array, "suffix_tokens channels"], StateLayerBase | None]:
            mixer_input, positional_embedding, mixer_state, length_without_padding = mixer_inputs
            return self.mixer(
                mixer_input,
                positional_embedding,
                mixer_state,
                return_updated_state=return_updated_state or return_activation_trace,
                length_without_padding=length_without_padding,
                forward_pass_config=forward_pass_config.mixer,
                keychain=mixer_keychain,
            )

        mixer_outputs, updated_state = call_vmapped(
            call_mixer,
            (normalized_mixer_inputs, positional_embeddings, state, lengths_without_padding),
        )
        if self.post_mixer_norm is not None:
            normalized_mixer_outputs = call_vmapped_twice(self.post_mixer_norm, mixer_outputs)
            mlp_inputs = inputs + normalized_mixer_outputs
        else:
            normalized_mixer_outputs = None
            mlp_inputs = inputs + mixer_outputs

        normalized_mlp_inputs = (
            call_vmapped_twice(self.pre_mlp_norm, mlp_inputs) if self.pre_mlp_norm is not None else mlp_inputs
        )
        mlp_outputs = self.mlp(
            normalized_mlp_inputs,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=forward_pass_config.mlp,
            keychain=mlp_keychain,
        )
        if self.post_mlp_norm is not None:
            normalized_mlp_outputs = call_vmapped_twice(self.post_mlp_norm, mlp_outputs)
            outputs = mlp_inputs + normalized_mlp_outputs
        else:
            normalized_mlp_outputs = None
            outputs = mlp_inputs + mlp_outputs

        if self.ple is not None and per_layer_input is not None:
            outputs = self.ple(outputs, per_layer_input)
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
