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
from .common import ForwardPassMode, LalamoModule, PositionalEmbeddingSelector
from .linear import LinearBase, LinearConfig
from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings
from .token_mixers import KVCacheLayer, StateLayerBase, StaticKVCacheLayer, TokenMixerBase, TokenMixerConfig
from .utils import vmap_twice

__all__ = [
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
    has_layer_scalar: bool = False
    kv_source_layer: int | None = None

    @property
    def rope_dim(self) -> int | None:
        return self.mixer_config.rope_dim

    def _init_ple(
        self, model_dim: int, *, empty: bool, key: PRNGKeyArray | None = None
    ) -> tuple[
        LinearBase | None,
        LinearBase | None,
        Normalization | None,
    ]:
        if self.ple_config is None:
            return None, None, None
        cfg = self.ple_config
        if empty:
            gate = cfg.linear_config.empty(model_dim, (cfg.ple_dim,), has_biases=False)
            projection = cfg.linear_config.empty(cfg.ple_dim, (model_dim,), has_biases=False)
        else:
            assert key is not None
            k1, k2 = jax.random.split(key)
            gate = cfg.linear_config.random_init(model_dim, (cfg.ple_dim,), has_biases=False, key=k1)
            projection = cfg.linear_config.random_init(cfg.ple_dim, (model_dim,), has_biases=False, key=k2)
        if empty:
            norm = cfg.norm_config.empty(model_dim)
        else:
            norm = cfg.norm_config.init(model_dim)
        return gate, projection, norm

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
        mixer = self.mixer_config.random_init(
            model_dim=model_dim,
            key=attention_key,
        )
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
        ple_gate, ple_projection, ple_norm = self._init_ple(model_dim, empty=False, key=ple_key)
        layer_scalar = None
        if self.has_layer_scalar:
            layer_scalar = jnp.ones((1,), dtype=jnp.bfloat16)
        return TransformerLayer(
            config=self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=mixer,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
            ple_gate=ple_gate,
            ple_projection=ple_projection,
            ple_norm=ple_norm,
            layer_scalar=layer_scalar,
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
        attention = self.mixer_config.empty(
            model_dim=model_dim,
        )
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
        ple_gate, ple_projection, ple_norm = self._init_ple(model_dim, empty=True)
        layer_scalar = None
        if self.has_layer_scalar:
            layer_scalar = dummy_array((1,), jnp.bfloat16)
        return TransformerLayer(
            config=self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=attention,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
            ple_gate=ple_gate,
            ple_projection=ple_projection,
            ple_norm=ple_norm,
            layer_scalar=layer_scalar,
        )


class TransformerLayer(LalamoModule[TransformerLayerConfig]):
    pre_mixer_norm: Normalization | None
    mixer: TokenMixerBase
    post_mixer_norm: Normalization | None
    pre_mlp_norm: Normalization | None
    mlp: MLPBase
    post_mlp_norm: Normalization | None
    ple_gate: LinearBase | None = None
    ple_projection: LinearBase | None = None
    ple_norm: Normalization | None = None
    layer_scalar: Float[Array, "1"] | None = None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.mixer.activation_precision

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return self.mixer.positional_embedding_selector

    def __post_init__(self) -> None:
        model_dim = self.pre_mixer_norm.input_dim if self.pre_mixer_norm is not None else self.mixer.model_dim
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

        normalized_mlp_inputs = (
            vmap_twice(self.pre_mlp_norm)(mlp_inputs) if self.pre_mlp_norm is not None else mlp_inputs
        )
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

        if self.ple_gate is not None and per_layer_input is not None:
            assert self.ple_projection is not None
            assert self.ple_norm is not None
            assert self.config.ple_config is not None
            ple_residual = outputs
            (ple_gated,) = vmap(vmap(self.ple_gate))(outputs)
            ple_gated = self.config.ple_config.activation(ple_gated)
            ple_gated = ple_gated * per_layer_input
            (ple_projected,) = vmap(vmap(self.ple_projection))(ple_gated)
            ple_normed = vmap_twice(self.ple_norm)(ple_projected)
            outputs = ple_residual + ple_normed

        if self.layer_scalar is not None:
            outputs = outputs * self.layer_scalar

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
        result: dict[str, ParameterTree | Array] = dict(
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
        if self.ple_gate is not None:
            result["ple_gate"] = self.ple_gate.export_weights()
        if self.ple_projection is not None:
            result["ple_projection"] = self.ple_projection.export_weights()
        if self.ple_norm is not None:
            result["ple_norm"] = self.ple_norm.export_weights()
        if self.layer_scalar is not None:
            result["layer_scalar"] = self.layer_scalar
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        if self.post_mixer_norm is not None:
            post_mixer_norm = self.post_mixer_norm.import_weights(require_tree(weights["post_mixer_norm"]))
        else:
            post_mixer_norm = None
        if self.post_mlp_norm is not None:
            post_mlp_norm = self.post_mlp_norm.import_weights(require_tree(weights["post_mlp_norm"]))
        else:
            post_mlp_norm = None
        if self.pre_mixer_norm is not None:
            pre_mixer_norm = self.pre_mixer_norm.import_weights(require_tree(weights["pre_mixer_norm"]))
        else:
            pre_mixer_norm = None
        if self.pre_mlp_norm is not None:
            pre_mlp_norm = self.pre_mlp_norm.import_weights(require_tree(weights["pre_mlp_norm"]))
        else:
            pre_mlp_norm = None
        ple_gate = None
        if self.ple_gate is not None:
            ple_gate = self.ple_gate.import_weights(require_tree(weights["ple_gate"]))
        ple_projection = None
        if self.ple_projection is not None:
            ple_projection = self.ple_projection.import_weights(require_tree(weights["ple_projection"]))
        ple_norm = None
        if self.ple_norm is not None:
            ple_norm = self.ple_norm.import_weights(require_tree(weights["ple_norm"]))
        layer_scalar = None
        if self.layer_scalar is not None:
            layer_scalar = weights.get("layer_scalar")
        return replace(
            self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=self.mixer.import_weights(require_tree(weights["mixer"])),
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=self.mlp.import_weights(require_tree(weights["mlp"])),
            post_mlp_norm=post_mlp_norm,
            ple_gate=ple_gate,
            ple_projection=ple_projection,
            ple_norm=ple_norm,
            layer_scalar=layer_scalar,
        )
