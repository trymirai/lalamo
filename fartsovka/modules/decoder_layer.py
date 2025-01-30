from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .activations import Activation
from .attention import AbstractAttention, AbstractAttentionConfig
from .common import FartsovkaModule, ModuleConfig, ParameterDict
from .kv_cache import KVCacheLayerSlice
from .mlp import AbstractMLP, AbstractMLPConfig
from .normalization import AbstractNormalization, AbstractNormalizationConfig
from .rope import PositionalEmbeddings

__all__ = [
    "DecoderLayer",
    "DecoderLayerConfig",
]


class DecoderLayerOutput(NamedTuple):
    output: Float[Array, "suffix_tokens channels"]
    kv_cache: KVCacheLayerSlice | None


class DecoderLayer[
    MLPNormType: AbstractNormalization,
    MLPType: AbstractMLP,
    AttentionNormType: AbstractNormalization,
    AttentionType: AbstractAttention,
](FartsovkaModule):
    model_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    activation: Activation = eqx.field(static=True)
    use_mlp_bias: bool = eqx.field(static=True)

    use_attention_qkv_bias: bool = eqx.field(static=True)
    use_attention_out_bias: bool = eqx.field(static=True)
    sliding_window_size: int | None = eqx.field(static=True)

    attention_norm: AttentionNormType
    attention: AttentionType
    mlp_norm: MLPNormType
    mlp: MLPType

    def __init__(
        self,
        *,
        attention_norm_config: AbstractNormalizationConfig[AttentionNormType],
        attention_config: AbstractAttentionConfig[AttentionType],
        mlp_norm_config: AbstractNormalizationConfig[MLPNormType],
        mlp_config: AbstractMLPConfig[MLPType],
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        activation: Activation,
        use_mlp_bias: bool,
        use_attention_qkv_bias: bool,
        use_attention_out_bias: bool,
        sliding_window_size: int | None,
        eps: float,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim

        self.activation = activation
        self.use_mlp_bias = use_mlp_bias

        self.use_attention_qkv_bias = use_attention_qkv_bias
        self.use_attention_out_bias = use_attention_out_bias
        self.sliding_window_size = sliding_window_size

        attention_key, mlp_key = jax.random.split(key)

        self.attention_norm = attention_norm_config(model_dim, eps)
        self.attention = attention_config(
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            sliding_window_size=sliding_window_size,
            use_qkv_bias=use_attention_qkv_bias,
            use_out_bias=use_attention_out_bias,
            key=attention_key,
        )
        self.mlp_norm = mlp_norm_config(model_dim, eps)
        self.mlp = mlp_config(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            use_bias=use_mlp_bias,
            activation=activation,
            key=mlp_key,
        )

    def __call__(
        self,
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> DecoderLayerOutput:
        residual = x
        x = vmap(self.attention_norm, in_axes=0)(x)
        attention_output = self.attention(x, positional_embeddings, kv_cache, mask, return_updated_kv_cache)
        x = residual + attention_output.attention_output
        updated_kv_cache = attention_output.kv_cache

        residual = x
        x = vmap(self.mlp_norm, in_axes=0)(x)
        x = residual + vmap(self.mlp, in_axes=0)(x)

        return DecoderLayerOutput(output=x, kv_cache=updated_kv_cache)

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            attention_norm=self.attention_norm.export_weights(),
            attention=self.attention.export_weights(),
            mlp_norm=self.mlp_norm.export_weights(),
            mlp=self.mlp.export_weights(),
        )


@dataclass
class DecoderLayerConfig[
    MLPNormType: AbstractNormalization,
    MLPType: AbstractMLP,
    AttentionNormType: AbstractNormalization,
    AttentionType: AbstractAttention,
](ModuleConfig[DecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]]):
    attention_norm_config: AbstractNormalizationConfig[AttentionNormType]
    attention_config: AbstractAttentionConfig[AttentionType]
    mlp_norm_config: AbstractNormalizationConfig[MLPNormType]
    mlp_config: AbstractMLPConfig[MLPType]

    def __call__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        activation: Activation,
        use_mlp_bias: bool,
        use_attention_qkv_bias: bool,
        use_attention_out_bias: bool,
        sliding_window_size: int | None,
        eps: float,
        key: PRNGKeyArray,
    ) -> DecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]:
        return DecoderLayer(
            attention_norm_config=self.attention_norm_config,
            attention_config=self.attention_config,
            mlp_norm_config=self.mlp_norm_config,
            mlp_config=self.mlp_config,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            activation=activation,
            use_mlp_bias=use_mlp_bias,
            use_attention_qkv_bias=use_attention_qkv_bias,
            use_attention_out_bias=use_attention_out_bias,
            sliding_window_size=sliding_window_size,
            eps=eps,
            key=key,
        )
