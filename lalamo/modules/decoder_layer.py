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

from .attention import Attention, AttentionConfig
from .common import AttentionType, ForwardPassMode, LalamoModule
from .kv_cache import KVCacheLayer, StaticKVCacheLayer
from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig
from .normalization import RMSNorm, RMSNormConfig
from .rope import PositionalEmbeddings
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
    kv_cache: KVCacheLayer | None

    mlp_inputs: Float[Array, "batch suffix_tokens channels"]
    pre_attention_norm: Float[Array, "batch suffix_tokens channels"]
    attention: Float[Array, "batch suffix_tokens channels"]
    post_attention_norm: Float[Array, "batch suffix_tokens channels"] | None
    pre_mlp_norm: Float[Array, "batch suffix_tokens channels"]
    mlp: Float[Array, "batch suffix_tokens channels"]
    post_mlp_norm: Float[Array, "batch suffix_tokens channels"] | None

    def export(self) -> ParameterTree:
        result = dict(
            inputs=self.inputs,
            positional_embeddings=self.positional_embeddings.export(),
            mlp_inputs=self.mlp_inputs,
            pre_attention_norm=self.pre_attention_norm,
            attention=self.attention,
            pre_mlp_norm=self.pre_mlp_norm,
            mlp=self.mlp,
        )
        if self.kv_cache is not None:
            result["kv_cache"] = self.kv_cache.export()
        if self.post_attention_norm is not None:
            result["post_attention_norm"] = self.post_attention_norm
        if self.post_mlp_norm is not None:
            result["post_mlp_norm"] = self.post_mlp_norm
        return result


class DecoderLayerResult(eqx.Module):
    outputs: Float[Array, "suffix_tokens channels"]
    updated_kv_cache: KVCacheLayer | None
    activation_trace: DecoderLayerActivationTrace | None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            outputs=self.outputs,
        )
        if self.updated_kv_cache is not None:
            result["updated_kv_cache"] = self.updated_kv_cache.export()
        if self.activation_trace is not None:
            result["activation_trace"] = self.activation_trace.export()
        return result


@dataclass(frozen=True)
class DecoderLayerConfig:
    pre_attention_norm_config: RMSNormConfig
    attention_config: AttentionConfig
    post_attention_norm_config: RMSNormConfig | None
    pre_mlp_norm_config: RMSNormConfig
    mlp_config: MLPConfig
    post_mlp_norm_config: RMSNormConfig | None

    def random_init(
        self,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        attention_scale: float | None,
        sliding_window_size: int | None,
        *,
        key: PRNGKeyArray,
    ) -> "DecoderLayer":
        attention_key, mlp_key = jax.random.split(key)
        pre_attention_norm = self.pre_attention_norm_config.init(model_dim)
        attention = self.attention_config.random_init(
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            is_causal=True,
            scale=attention_scale,
            sliding_window_size=sliding_window_size,
            key=attention_key,
        )
        if self.post_attention_norm_config is not None:
            post_attention_norm = self.post_attention_norm_config.init(model_dim)
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
            pre_attention_norm=pre_attention_norm,
            attention=attention,
            post_attention_norm=post_attention_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
        )

    def empty(
        self,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        attention_scale: float | None,
        sliding_window_size: int | None,
    ) -> "DecoderLayer":
        pre_attention_norm = self.pre_attention_norm_config.empty(model_dim)
        attention = self.attention_config.empty(
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            is_causal=True,
            scale=attention_scale,
            sliding_window_size=sliding_window_size,
        )
        if self.post_attention_norm_config is not None:
            post_attention_norm = self.post_attention_norm_config.empty(model_dim)
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
            pre_attention_norm=pre_attention_norm,
            attention=attention,
            post_attention_norm=post_attention_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
        )


class DecoderLayer(LalamoModule[DecoderLayerConfig]):
    pre_attention_norm: RMSNorm
    attention: Attention
    post_attention_norm: RMSNorm | None
    pre_mlp_norm: RMSNorm
    mlp: MLPBase
    post_mlp_norm: RMSNorm | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.attention.activation_precision

    @property
    def attention_type(self) -> AttentionType:
        return self.attention.attention_type

    def __post_init__(self) -> None:
        model_dim = self.pre_attention_norm.input_dim
        if self.attention.model_dim != model_dim:
            raise ValueError(
                f"Attention model dim {self.attention.model_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )
        if self.post_attention_norm is not None and self.post_attention_norm.input_dim != model_dim:
            raise ValueError(
                f"Post attention normalization dim {self.post_attention_norm.input_dim} does not match"
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
        kv_cache: KVCacheLayer | None = None,
        return_updated_kv_cache: bool = False,
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
        normalized_attention_inputs = vmap_twice(self.pre_attention_norm)(inputs)
        batched_attention_fn = vmap(partial(self.attention, return_updated_kv_cache=return_updated_kv_cache))
        attention_outputs, updated_kv_cache = batched_attention_fn(
            normalized_attention_inputs,
            positional_embeddings,
            kv_cache=kv_cache,
            length_without_padding=lengths_without_padding,
        )
        if self.post_attention_norm is not None:
            normalized_attention_outputs = vmap_twice(self.post_attention_norm)(attention_outputs)
            mlp_inputs = inputs + normalized_attention_outputs
        else:
            normalized_attention_outputs = None
            mlp_inputs = inputs + attention_outputs

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
                kv_cache=kv_cache,
                pre_attention_norm=normalized_attention_inputs,
                attention=attention_outputs,
                post_attention_norm=normalized_attention_outputs,
                mlp_inputs=mlp_inputs,
                pre_mlp_norm=normalized_mlp_inputs,
                mlp=mlp_outputs,
                post_mlp_norm=normalized_mlp_outputs,
            )
        else:
            activation_trace = None

        return DecoderLayerResult(
            outputs=outputs,
            updated_kv_cache=updated_kv_cache,
            activation_trace=activation_trace,
        )

    def init_static_kv_cache(self, batch_size: int, capacity: int) -> StaticKVCacheLayer:
        return jax.tree.map(
            lambda array: jnp.repeat(array[None, ...], batch_size, axis=0),
            self.attention.init_static_kv_cache(capacity),
        )

    def export_weights(self) -> ParameterTree:
        result = dict(
            pre_attention_norm=self.pre_attention_norm.export_weights(),
            attention=self.attention.export_weights(),
            pre_mlp_norm=self.pre_mlp_norm.export_weights(),
            mlp=self.mlp.export_weights(),
        )
        if self.post_attention_norm is not None:
            result["post_attention_norm"] = self.post_attention_norm.export_weights()
        if self.post_mlp_norm is not None:
            result["post_mlp_norm"] = self.post_mlp_norm.export_weights()
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["pre_attention_norm"], Mapping)
        assert isinstance(weights["attention"], Mapping)
        assert isinstance(weights["mlp"], Mapping)
        assert isinstance(weights["pre_mlp_norm"], Mapping)

        if self.post_attention_norm is not None:
            assert isinstance(weights["post_attention_norm"], Mapping)
            post_attention_norm = self.post_attention_norm.import_weights(
                weights["post_attention_norm"],
            )
        else:
            post_attention_norm = None
        if self.post_mlp_norm is not None:
            assert isinstance(weights["post_mlp_norm"], Mapping)
            post_mlp_norm = self.post_mlp_norm.import_weights(weights["post_mlp_norm"])
        else:
            post_mlp_norm = None
        return replace(
            self,
            pre_attention_norm=self.pre_attention_norm.import_weights(weights["pre_attention_norm"]),
            attention=self.attention.import_weights(weights["attention"]),
            post_attention_norm=post_attention_norm,
            pre_mlp_norm=self.pre_mlp_norm.import_weights(weights["pre_mlp_norm"]),
            mlp=self.mlp.import_weights(weights["mlp"]),
            post_mlp_norm=post_mlp_norm,
        )
