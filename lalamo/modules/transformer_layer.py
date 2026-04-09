from dataclasses import dataclass, replace
from functools import partial
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_mapping, require_tree

from .activations import Activation
from .common import ForwardPassMode, LalamoModule
from .linear import LinearBase, LinearConfig
from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPEConfig
from .token_mixers import KVCacheLayer, StateLayerBase, StaticKVCacheLayer, TokenMixerBase, TokenMixerConfig
from .utils import vmap_twice

__all__ = [
    "PLELayer",
    "PLELayerConfig",
    "TransformerLayer",
    "TransformerLayerActivationTrace",
    "TransformerLayerConfig",
    "TransformerLayerForwardPassConfig",
    "TransformerLayerResult",
]


type TransformerLayerForwardPassConfig = MLPForwardPassConfig


class TransformerLayerActivationTrace(eqx.Module):
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

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            inputs=self.inputs,
            mlp_inputs=self.mlp_inputs,
            pre_mixer_norm=self.pre_mixer_norm,
            mixer=self.mixer,
            pre_mlp_norm=self.pre_mlp_norm,
            mlp=self.mlp,
        )
        if self.positional_embeddings is not None:
            result["positional_embeddings"] = self.positional_embeddings.export()
        if self.state is not None:
            result["state"] = self.state.export()
        if self.post_mixer_norm is not None:
            result["post_mixer_norm"] = self.post_mixer_norm
        if self.post_mlp_norm is not None:
            result["post_mlp_norm"] = self.post_mlp_norm
        return result


class TransformerLayerResult(eqx.Module):
    outputs: Float[Array, "batch tokens channels"]
    updated_state: KVCacheLayer | None
    activation_trace: TransformerLayerActivationTrace | None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            outputs=self.outputs,
        )
        if self.updated_state is not None:
            result["updated_state"] = self.updated_state.export()
        if self.activation_trace is not None:
            result["activation_trace"] = self.activation_trace.export()
        return result


@dataclass(frozen=True)
class PLELayerConfig:
    linear_config: LinearConfig
    norm_config: NormalizationConfig
    ple_dim: int
    activation: Activation
    has_layer_scalar: bool

    def init(self, model_dim: int, *, key: PRNGKeyArray) -> "PLELayer":
        k1, k2 = jax.random.split(key)
        gate = self.linear_config.random_init(model_dim, (self.ple_dim,), has_biases=False, key=k1)
        projection = self.linear_config.random_init(self.ple_dim, (model_dim,), has_biases=False, key=k2)
        norm = self.norm_config.init(model_dim)
        if self.has_layer_scalar:
            layer_scalar = jnp.ones((1,), dtype=jnp.bfloat16)
        else:
            layer_scalar = None
        return PLELayer(config=self, gate=gate, projection=projection, norm=norm, layer_scalar=layer_scalar)

    def empty(self, model_dim: int) -> "PLELayer":
        gate = self.linear_config.empty(model_dim, (self.ple_dim,), has_biases=False)
        projection = self.linear_config.empty(self.ple_dim, (model_dim,), has_biases=False)
        norm = self.norm_config.empty(model_dim)
        if self.has_layer_scalar:
            layer_scalar = dummy_array((1,), jnp.bfloat16)
        else:
            layer_scalar = None
        return PLELayer(config=self, gate=gate, projection=projection, norm=norm, layer_scalar=layer_scalar)


class PLELayer(LalamoModule[PLELayerConfig]):
    gate: LinearBase
    projection: LinearBase
    norm: Normalization
    layer_scalar: Float[Array, "1"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.gate.activation_precision

    def __call__(
        self,
        outputs: Float[Array, "batch suffix_tokens channels"],
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        if per_layer_input is not None:
            (ple_gated,) = vmap(vmap(self.gate))(outputs)
            ple_gated = self.config.activation(ple_gated)
            ple_gated = ple_gated * per_layer_input
            (ple_projected,) = vmap(vmap(self.projection))(ple_gated)
            ple_normed = vmap_twice(self.norm)(ple_projected)
            outputs = outputs + ple_normed
        if self.layer_scalar is not None:
            outputs = outputs * self.layer_scalar
        return outputs

    def export_weights(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = {
            "gate": self.gate.export_weights(),
            "projection": self.projection.export_weights(),
            "norm": self.norm.export_weights(),
        }
        if self.layer_scalar is not None:
            result["layer_scalar"] = self.layer_scalar
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        if self.layer_scalar is not None:
            layer_scalar = weights["layer_scalar"]
        else:
            layer_scalar = None
        return replace(
            self,
            gate=self.gate.import_weights(require_tree(weights["gate"])),
            projection=self.projection.import_weights(require_tree(weights["projection"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
            layer_scalar=layer_scalar,
        )


@dataclass(frozen=True)
class TransformerLayerConfig:
    pre_mixer_norm_config: NormalizationConfig | None
    mixer_config: TokenMixerConfig
    post_mixer_norm_config: NormalizationConfig | None
    pre_mlp_norm_config: NormalizationConfig
    mlp_config: MLPConfig
    post_mlp_norm_config: NormalizationConfig | None
    hidden_dim: int | None = None
    ple_config: PLELayerConfig | None = None
    kv_source_layer: int | None = None
    rope_config: RoPEConfig | None = None

    def random_init(
        self,
        model_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "TransformerLayer":
        attention_key, mlp_key, ple_key = jax.random.split(key, 3)
        if self.pre_mixer_norm_config is not None:
            pre_mixer_norm = self.pre_mixer_norm_config.init(model_dim)
        else:
            pre_mixer_norm = None
        mixer = self.mixer_config.random_init(model_dim=model_dim, key=attention_key)
        if self.post_mixer_norm_config is not None:
            post_mixer_norm = self.post_mixer_norm_config.init(model_dim)
        else:
            post_mixer_norm = None
        pre_mlp_norm = self.pre_mlp_norm_config.init(model_dim)
        mlp = self.mlp_config.random_init(model_dim, hidden_dim, key=mlp_key)
        if self.post_mlp_norm_config is not None:
            post_mlp_norm = self.post_mlp_norm_config.init(model_dim)
        else:
            post_mlp_norm = None
        if self.ple_config is not None:
            ple = self.ple_config.init(model_dim, key=ple_key)
        else:
            ple = None
        return TransformerLayer(
            config=self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=mixer,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
            ple=ple,
        )

    def empty(
        self,
        model_dim: int,
        hidden_dim: int,
    ) -> "TransformerLayer":
        if self.pre_mixer_norm_config is not None:
            pre_mixer_norm = self.pre_mixer_norm_config.empty(model_dim)
        else:
            pre_mixer_norm = None
        mixer = self.mixer_config.empty(model_dim=model_dim)
        if self.post_mixer_norm_config is not None:
            post_mixer_norm = self.post_mixer_norm_config.empty(model_dim)
        else:
            post_mixer_norm = None
        if self.pre_mlp_norm_config is not None:
            pre_mlp_norm = self.pre_mlp_norm_config.empty(model_dim)
        else:
            pre_mlp_norm = None
        mlp = self.mlp_config.empty(model_dim, hidden_dim)
        if self.post_mlp_norm_config is not None:
            post_mlp_norm = self.post_mlp_norm_config.empty(model_dim)
        else:
            post_mlp_norm = None
        if self.ple_config is not None:
            ple = self.ple_config.empty(model_dim)
        else:
            ple = None
        return TransformerLayer(
            config=self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=mixer,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
            ple=ple,
        )


class TransformerLayer(LalamoModule[TransformerLayerConfig]):
    pre_mixer_norm: Normalization | None
    mixer: TokenMixerBase
    post_mixer_norm: Normalization | None
    pre_mlp_norm: Normalization | None
    mlp: MLPBase
    post_mlp_norm: Normalization | None
    ple: PLELayer | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.mixer.activation_precision

    def __post_init__(self) -> None:
        if self.pre_mixer_norm is not None:
            model_dim = self.pre_mixer_norm.input_dim
        else:
            model_dim = self.mixer.model_dim
        if self.mixer.model_dim != model_dim:
            raise ValueError(
                f"Attention model dim {self.mixer.model_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )
        if self.post_mixer_norm is not None and self.post_mixer_norm.input_dim != model_dim:
            raise ValueError(
                f"Post mixer normalization dim {self.post_mixer_norm.input_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )
        if self.pre_mlp_norm and self.pre_mlp_norm.input_dim != model_dim:
            raise ValueError(
                f"Pre MLP normalization dim {self.pre_mlp_norm.input_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )
        if self.mlp.model_dim != model_dim:
            raise ValueError(
                f"MLP up projection dim {self.mlp.model_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )

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
        forward_pass_config: TransformerLayerForwardPassConfig | None = None,
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None = None,
    ) -> TransformerLayerResult:
        if inputs.ndim != 3:
            raise ValueError(
                f"Inputs to decoder layers must be a 3D arrays of size (batch_size, sequence_length, hidden_dim),"
                f" got {inputs.shape}",
            )
        if self.pre_mixer_norm is not None:
            normalized_mixer_inputs = vmap_twice(self.pre_mixer_norm)(inputs)
        else:
            normalized_mixer_inputs = inputs

        batched_mixer_fn = vmap(
            partial(self.mixer, return_updated_state=return_updated_state or return_activation_trace),
        )
        mixer_outputs, updated_state = batched_mixer_fn(
            normalized_mixer_inputs,
            positional_embeddings,
            state=state,
            length_without_padding=lengths_without_padding,
        )
        if self.post_mixer_norm is not None:
            normalized_mixer_outputs = vmap_twice(self.post_mixer_norm)(mixer_outputs)
            mlp_inputs = inputs + normalized_mixer_outputs
        else:
            normalized_mixer_outputs = None
            mlp_inputs = inputs + mixer_outputs

        if self.pre_mlp_norm is not None:
            normalized_mlp_inputs = vmap_twice(self.pre_mlp_norm)(mlp_inputs)
        else:
            normalized_mlp_inputs = mlp_inputs
        mlp_outputs = self.mlp(
            normalized_mlp_inputs,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=forward_pass_config,
        )
        if self.post_mlp_norm is not None:
            normalized_mlp_outputs = vmap_twice(self.post_mlp_norm)(mlp_outputs)
            outputs = mlp_inputs + normalized_mlp_outputs
        else:
            normalized_mlp_outputs = None
            outputs = mlp_inputs + mlp_outputs

        if self.ple is not None:
            outputs = self.ple(outputs, per_layer_input)

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

    def init_static_state(self, batch_size: int, capacity: int) -> StaticKVCacheLayer:
        return jax.tree.map(
            lambda array: jnp.repeat(array[None, ...], batch_size, axis=0),
            self.mixer.init_static_state(capacity),
        )

    def export_weights(self) -> ParameterTree:
        result = dict(
            mixer=self.mixer.export_weights(),
            mlp=self.mlp.export_weights(),
        )
        if self.pre_mixer_norm is not None:
            result["pre_mixer_norm"] = self.pre_mixer_norm.export_weights()
        if self.post_mixer_norm is not None:
            result["post_mixer_norm"] = self.post_mixer_norm.export_weights()
        if self.post_mlp_norm is not None:
            result["post_mlp_norm"] = self.post_mlp_norm.export_weights()
        if self.pre_mlp_norm is not None:
            result["pre_mlp_norm"] = self.pre_mlp_norm.export_weights()
        if self.ple is not None:
            result["ple"] = self.ple.export_weights()
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        pre_mixer_norm = (
            self.pre_mixer_norm.import_weights(require_tree(weights["pre_mixer_norm"]))
            if self.pre_mixer_norm is not None
            else None
        )
        post_mixer_norm = (
            self.post_mixer_norm.import_weights(require_tree(weights["post_mixer_norm"]))
            if self.post_mixer_norm is not None
            else None
        )
        pre_mlp_norm = (
            self.pre_mlp_norm.import_weights(require_tree(weights["pre_mlp_norm"]))
            if self.pre_mlp_norm is not None
            else None
        )
        post_mlp_norm = (
            self.post_mlp_norm.import_weights(require_tree(weights["post_mlp_norm"]))
            if self.post_mlp_norm is not None
            else None
        )
        if self.ple is not None:
            ple = self.ple.import_weights(require_tree(weights["ple"]))
        else:
            ple = None
        return replace(
            self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=self.mixer.import_weights(require_tree(weights["mixer"])),
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=self.mlp.import_weights(require_tree(weights["mlp"])),
            post_mlp_norm=post_mlp_norm,
            ple=ple,
        )
