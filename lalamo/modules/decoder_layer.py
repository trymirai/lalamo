from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.state.common import StateLayerBase
from lalamo.modules.token_mixers import TokenMixerBase, TokenMixerConfig

from .common import ForwardPassMode, LalamoModule, PositionalEmbeddingSelector
from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import RMSNorm, RMSNormConfig
from .rope import PositionalEmbeddings
from .state import KVCacheLayer, StaticKVCacheLayer
from .utils import vmap_twice

__all__ = [
    "DecoderLayer",
    "DecoderLayerActivationTrace",
    "DecoderLayerConfig",
    "DecoderLayerForwardPassConfig",
    "DecoderLayerResult",
]


type DecoderLayerForwardPassConfig = MLPForwardPassConfig


class DecoderLayerActivationTrace(eqx.Module):
    inputs: Float[Array, "batch suffix_tokens channels"]
    positional_embeddings: PositionalEmbeddings
    state: StateLayerBase | None

    mlp_inputs: Float[Array, "batch suffix_tokens channels"]
    pre_mixer_norm: Float[Array, "batch suffix_tokens channels"]
    mixer: Float[Array, "batch suffix_tokens channels"]
    post_mixer_norm: Float[Array, "batch suffix_tokens channels"] | None
    pre_mlp_norm: Float[Array, "batch suffix_tokens channels"]
    mlp: Float[Array, "batch suffix_tokens channels"]
    post_mlp_norm: Float[Array, "batch suffix_tokens channels"] | None

    def export(self) -> ParameterTree:
        result = dict(
            inputs=self.inputs,
            positional_embeddings=self.positional_embeddings.export(),
            mlp_inputs=self.mlp_inputs,
            pre_mixer_norm=self.pre_mixer_norm,
            mixer=self.mixer,
            pre_mlp_norm=self.pre_mlp_norm,
            mlp=self.mlp,
        )
        if self.state is not None:
            result["kv_cache"] = self.state.export()
        if self.post_mixer_norm is not None:
            result["post_mixer_norm"] = self.post_mixer_norm
        if self.post_mlp_norm is not None:
            result["post_mlp_norm"] = self.post_mlp_norm
        return result


class DecoderLayerResult(eqx.Module):
    outputs: Float[Array, "suffix_tokens channels"]
    updated_state: KVCacheLayer | None
    activation_trace: DecoderLayerActivationTrace | None

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
class DecoderLayerConfig:
    pre_mixer_norm_config: RMSNormConfig
    mixer_config: TokenMixerConfig
    post_mixer_norm_config: RMSNormConfig | None
    pre_mlp_norm_config: RMSNormConfig
    mlp_config: MLPConfig
    post_mlp_norm_config: RMSNormConfig | None

    @property
    def rope_dim(self) -> int:
        return self.mixer_config.rope_dim

    def random_init(
        self,
        model_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "DecoderLayer":
        attention_key, mlp_key = jax.random.split(key)
        pre_attention_norm = self.pre_mixer_norm_config.init(model_dim)
        mixer = self.mixer_config.random_init(
            model_dim=model_dim,
            key=attention_key,
        )
        if self.post_mixer_norm_config is not None:
            post_attention_norm = self.post_mixer_norm_config.init(model_dim)
        else:
            post_attention_norm = None
        pre_mlp_norm = self.pre_mlp_norm_config.init(model_dim)
        mlp = self.mlp_config.random_init(model_dim, hidden_dim, key=mlp_key)
        if self.post_mlp_norm_config is not None:
            post_mlp_norm = self.post_mlp_norm_config.init(model_dim)
        else:
            post_mlp_norm = None
        return DecoderLayer(
            config=self,
            pre_mixer_norm=pre_attention_norm,
            mixer=mixer,
            post_mixer_norm=post_attention_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
        )

    def empty(
        self,
        model_dim: int,
        hidden_dim: int,
    ) -> "DecoderLayer":
        pre_attention_norm = self.pre_mixer_norm_config.empty(model_dim)
        attention = self.mixer_config.empty(
            model_dim=model_dim,
        )
        if self.post_mixer_norm_config is not None:
            post_attention_norm = self.post_mixer_norm_config.empty(model_dim)
        else:
            post_attention_norm = None
        pre_mlp_norm = self.pre_mlp_norm_config.empty(model_dim)
        mlp = self.mlp_config.empty(model_dim, hidden_dim)
        if self.post_mlp_norm_config is not None:
            post_mlp_norm = self.post_mlp_norm_config.empty(model_dim)
        else:
            post_mlp_norm = None
        return DecoderLayer(
            config=self,
            pre_mixer_norm=pre_attention_norm,
            mixer=attention,
            post_mixer_norm=post_attention_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
        )


class DecoderLayer(LalamoModule[DecoderLayerConfig]):
    pre_mixer_norm: RMSNorm
    mixer: TokenMixerBase
    post_mixer_norm: RMSNorm | None
    pre_mlp_norm: RMSNorm
    mlp: MLPBase
    post_mlp_norm: RMSNorm | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.mixer.activation_precision

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return self.mixer.positional_embedding_selector

    def __post_init__(self) -> None:
        model_dim = self.pre_mixer_norm.input_dim
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
        positional_embeddings: PositionalEmbeddings,
        state: StateLayerBase | None = None,
        return_updated_state: bool = False,
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: DecoderLayerForwardPassConfig | None = None,
    ) -> DecoderLayerResult:
        if inputs.ndim != 3:
            raise ValueError(
                f"Inputs to decoder layers must be a 3D arrays of size (batch_size, sequence_length, hidden_dim),"
                f" got {inputs.shape}",
            )
        normalized_mixer_inputs = vmap_twice(self.pre_mixer_norm)(inputs)
        batched_mixer_fn = vmap(partial(self.mixer, return_updated_kv_cache=return_updated_state))
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
            activation_trace = DecoderLayerActivationTrace(
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

        return DecoderLayerResult(
            outputs=outputs,
            updated_state=updated_state,
            activation_trace=activation_trace,
        )

    def init_static_kv_cache(self, batch_size: int, capacity: int) -> StaticKVCacheLayer:
        return jax.tree.map(
            lambda array: jnp.repeat(array[None, ...], batch_size, axis=0),
            self.mixer.init_static_state(capacity),
        )

    def export_weights(self) -> ParameterTree:
        result = dict(
            pre_mixer_norm=self.pre_mixer_norm.export_weights(),
            mixer=self.mixer.export_weights(),
            pre_mlp_norm=self.pre_mlp_norm.export_weights(),
            mlp=self.mlp.export_weights(),
        )
        if self.post_mixer_norm is not None:
            result["post_mixer_norm"] = self.post_mixer_norm.export_weights()
        if self.post_mlp_norm is not None:
            result["post_mlp_norm"] = self.post_mlp_norm.export_weights()
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["pre_mixer_norm"], Mapping)
        assert isinstance(weights["mixer"], Mapping)
        assert isinstance(weights["mlp"], Mapping)
        assert isinstance(weights["pre_mlp_norm"], Mapping)

        if self.post_mixer_norm is not None:
            assert isinstance(weights["post_mixer_norm"], Mapping)
            post_mixer_norm = self.post_mixer_norm.import_weights(
                weights["post_mixer_norm"],
            )
        else:
            post_mixer_norm = None
        if self.post_mlp_norm is not None:
            assert isinstance(weights["post_mlp_norm"], Mapping)
            post_mlp_norm = self.post_mlp_norm.import_weights(weights["post_mlp_norm"])
        else:
            post_mlp_norm = None
        return replace(
            self,
            pre_mixer_norm=self.pre_mixer_norm.import_weights(weights["pre_mixer_norm"]),
            mixer=self.mixer.import_weights(weights["mixer"]),
            post_mixer_norm=post_mixer_norm,
            pre_mlp_norm=self.pre_mlp_norm.import_weights(weights["pre_mlp_norm"]),
            mlp=self.mlp.import_weights(weights["mlp"]),
            post_mlp_norm=post_mlp_norm,
        )
