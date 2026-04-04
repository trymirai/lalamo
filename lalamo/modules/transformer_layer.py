from dataclasses import dataclass, replace
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.arrays import ArrayForwardPassConfig, StochasticQuantize
from lalamo.common import ParameterTree

from .common import ForwardPassMode, Initializer, LalamoModule, PositionalEmbeddingSelector
from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import Normalization, NormalizationConfig, NormalizationForwardPassConfig
from .rope import PositionalEmbeddings
from .token_mixers import KVCacheLayer, StateLayerBase, StaticKVCacheLayer, TokenMixerBase, TokenMixerConfig
from .token_mixers.common import MixerForwardPassConfig
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


class TransformerLayerForwardPassConfig(eqx.Module):
    mixer: MixerForwardPassConfig = MixerForwardPassConfig()
    mlp: MLPForwardPassConfig = MLPForwardPassConfig()
    normalization: NormalizationForwardPassConfig = NormalizationForwardPassConfig()

    @staticmethod
    def init(
        arrays: ArrayForwardPassConfig | None = None,
        *,
        stochastic_quantize_key: jax.Array | None = None,
        moe_chunk_size_ratio: float = 0.2,
        normalization: NormalizationForwardPassConfig = NormalizationForwardPassConfig(),  # noqa: B008
    ) -> "TransformerLayerForwardPassConfig":
        if arrays is not None and stochastic_quantize_key is not None:
            raise ValueError("Pass either arrays or stochastic_quantize_key, not both.")
        if stochastic_quantize_key is not None:
            arrays = ArrayForwardPassConfig(quantize=StochasticQuantize(stochastic_quantize_key))
        if arrays is None:
            arrays = ArrayForwardPassConfig()

        def split_array_forward_pass_config(num: int) -> tuple[ArrayForwardPassConfig, ...]:
            match arrays.quantize:
                case StochasticQuantize(key=key):
                    return tuple(
                        replace(arrays, quantize=StochasticQuantize(subkey)) for subkey in jax.random.split(key, num)
                    )
                case _:
                    return (arrays,) * num

        mixer_in_arrays, mixer_gate_arrays, mixer_out_arrays = split_array_forward_pass_config(3)
        mlp_up_arrays, mlp_down_arrays, mlp_router_arrays, mlp_gate_arrays = split_array_forward_pass_config(4)
        return TransformerLayerForwardPassConfig(
            mixer=MixerForwardPassConfig(
                in_arrays=mixer_in_arrays,
                gate_arrays=mixer_gate_arrays,
                out_arrays=mixer_out_arrays,
            ),
            mlp=MLPForwardPassConfig(
                moe_chunk_size_ratio=moe_chunk_size_ratio,
                up_arrays=mlp_up_arrays,
                down_arrays=mlp_down_arrays,
                router_arrays=mlp_router_arrays,
                gate_arrays=mlp_gate_arrays,
            ),
            normalization=normalization,
        )


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

    def init(self, model_dim: int, *, key: PRNGKeyArray) -> "PLELayer":
        k1, k2 = jax.random.split(key)
        gate = self.linear_config.random_init(model_dim, (self.ple_dim,), has_biases=False, key=k1)
        projection = self.linear_config.random_init(self.ple_dim, (model_dim,), has_biases=False, key=k2)
        norm = self.norm_config.init(model_dim)
        return PLELayer(config=self, gate=gate, projection=projection, norm=norm)

    def empty(self, model_dim: int) -> "PLELayer":
        gate = self.linear_config.empty(model_dim, (self.ple_dim,), has_biases=False)
        projection = self.linear_config.empty(self.ple_dim, (model_dim,), has_biases=False)
        norm = self.norm_config.empty(model_dim)
        return PLELayer(config=self, gate=gate, projection=projection, norm=norm)


class PLELayer(LalamoModule[PLELayerConfig]):
    gate: LinearBase
    projection: LinearBase
    norm: Normalization

    @property
    def activation_precision(self) -> DTypeLike:
        return self.gate.activation_precision

    def __call__(
        self,
        outputs: Float[Array, "batch suffix_tokens channels"],
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"],
    ) -> Float[Array, "batch suffix_tokens channels"]:
        (ple_gated,) = vmap(vmap(self.gate))(outputs)
        ple_gated = self.config.activation(ple_gated)
        ple_gated = ple_gated * per_layer_input
        (ple_projected,) = vmap(vmap(self.projection))(ple_gated)
        ple_normed = vmap_twice(self.norm)(ple_projected)
        return outputs + ple_normed

    def export_weights(self) -> ParameterTree:
        return {
            "gate": self.gate.export_weights(),
            "projection": self.projection.export_weights(),
            "norm": self.norm.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        return replace(
            self,
            gate=self.gate.import_weights(require_tree(weights["gate"])),
            projection=self.projection.import_weights(require_tree(weights["projection"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
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
    def activation_precision(self) -> DTypeLike:
        return self.mlp.activation_precision

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
        forward_pass_config: TransformerLayerForwardPassConfig | None = None,
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
    ) -> TransformerLayerResult:
        if inputs.ndim != 3:
            raise ValueError(
                f"Inputs to decoder layers must be a 3D arrays of size (batch_size, sequence_length, hidden_dim),"
                f" got {inputs.shape}",
            )
        fpc = forward_pass_config or TransformerLayerForwardPassConfig()

        def apply_norm(
            norm: Normalization, x: Float[Array, "batch tokens channels"]
        ) -> Float[Array, "batch tokens channels"]:
            return vmap_twice(partial(norm, forward_pass_config=fpc.normalization))(x)

        if self.pre_mixer_norm is not None:
            normalized_mixer_inputs = apply_norm(self.pre_mixer_norm, inputs)
        else:
            normalized_mixer_inputs = inputs

        batched_mixer_fn = vmap(
            partial(
                self.mixer,
                return_updated_state=return_updated_state or return_activation_trace,
                forward_pass_config=fpc.mixer,
            ),
        )
        mixer_outputs, updated_state = batched_mixer_fn(
            normalized_mixer_inputs,
            positional_embeddings,
            state=state,
            length_without_padding=lengths_without_padding,
        )
        if self.post_mixer_norm is not None:
            normalized_mixer_outputs = apply_norm(self.post_mixer_norm, mixer_outputs)
            mlp_inputs = inputs + normalized_mixer_outputs
        else:
            normalized_mixer_outputs = None
            mlp_inputs = inputs + mixer_outputs

        normalized_mlp_inputs = (
            apply_norm(self.pre_mlp_norm, mlp_inputs) if self.pre_mlp_norm is not None else mlp_inputs
        )
        mlp_outputs = self.mlp(
            normalized_mlp_inputs,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=fpc.mlp,
        )
        if self.post_mlp_norm is not None:
            normalized_mlp_outputs = apply_norm(self.post_mlp_norm, mlp_outputs)
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

    def init_static_state(self, batch_size: int, capacity: int) -> StaticKVCacheLayer:
        return jax.tree.map(
            lambda array: jnp.repeat(array[None, ...], batch_size, axis=0),
            self.mixer.init_static_state(capacity),
        )
