from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree

from .common import ForwardPassMode, LalamoModule, PositionalEmbeddingSelector
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
class TransformerLayerConfig:
    pre_mixer_norm_config: NormalizationConfig | None
    mixer_config: TokenMixerConfig
    post_mixer_norm_config: NormalizationConfig | None
    pre_mlp_norm_config: NormalizationConfig
    mlp_config: MLPConfig
    post_mlp_norm_config: NormalizationConfig | None

    @property
    def rope_dim(self) -> int | None:
        return self.mixer_config.rope_dim

    def random_init(
        self,
        model_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "TransformerLayer":
        attention_key, mlp_key = jax.random.split(key)
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
        return TransformerLayer(
            config=self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=mixer,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
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
        pre_mlp_norm = self.pre_mlp_norm_config.empty(model_dim)
        mlp = self.mlp_config.empty(model_dim, hidden_dim)
        if self.post_mlp_norm_config is not None:
            post_mlp_norm = self.post_mlp_norm_config.empty(model_dim)
        else:
            post_mlp_norm = None
        return TransformerLayer(
            config=self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=attention,
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
        )


class TransformerLayer(LalamoModule[TransformerLayerConfig]):
    pre_mixer_norm: Normalization | None
    mixer: TokenMixerBase
    post_mixer_norm: Normalization | None
    pre_mlp_norm: Normalization
    mlp: MLPBase
    post_mlp_norm: Normalization | None

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
        if self.pre_mlp_norm.input_dim != model_dim:
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

        normalized_mlp_inputs = vmap_twice(self.pre_mlp_norm)(mlp_inputs)
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
            pre_mlp_norm=self.pre_mlp_norm.export_weights(),
            mlp=self.mlp.export_weights(),
        )
        if self.pre_mixer_norm is not None:
            result["pre_mixer_norm"] = self.pre_mixer_norm.export_weights()
        if self.post_mixer_norm is not None:
            result["post_mixer_norm"] = self.post_mixer_norm.export_weights()
        if self.post_mlp_norm is not None:
            result["post_mlp_norm"] = self.post_mlp_norm.export_weights()
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
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
        return replace(
            self,
            pre_mixer_norm=pre_mixer_norm,
            mixer=self.mixer.import_weights(require_tree(weights["mixer"])),
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=self.pre_mlp_norm.import_weights(require_tree(weights["pre_mlp_norm"])),
            mlp=self.mlp.import_weights(require_tree(weights["mlp"])),
            post_mlp_norm=post_mlp_norm,
        )
