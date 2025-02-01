from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .activations import Activation
from .attention import AttentionBase, AbstractAttentionConfig, AttentionConfigType
from .common import FartsovkaModule, ParameterDict, register_config_union
from .kv_cache import KVCacheLayerSlice
from .mlp import AbstractMLP, AbstractMLPConfig, MLPConfigType
from .normalization import AbstractNormalization, AbstractNormalizationConfig, NormalizationConfigType
from .rope import PositionalEmbeddings

__all__ = [
    "AbstractDecoderLayer",
    "AbstractDecoderLayerConfig",
    "DecoderLayerConfigType",
    "DecoderLayerOutput",
    "PreNormDecoderLayer",
    "PreNormDecoderLayerConfig",
]


class DecoderLayerOutput(NamedTuple):
    output: Float[Array, "suffix_tokens channels"]
    kv_cache: KVCacheLayerSlice | None


class AbstractDecoderLayer(FartsovkaModule):
    def __call__(
        self,
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> DecoderLayerOutput:
        raise NotImplementedError


@dataclass
class AbstractDecoderLayerConfig[DecoderLayerType: AbstractDecoderLayer]:
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
        attention_scale: float | None,
        attention_logit_soft_cap: float | None,
        sliding_window_size: int | None,
        eps: float,
        key: PRNGKeyArray,
    ) -> DecoderLayerType:
        raise NotImplementedError


class PreNormDecoderLayer[
    MLPNormType: AbstractNormalization,
    MLPType: AbstractMLP,
    AttentionNormType: AbstractNormalization,
    AttentionType: AttentionBase,
](AbstractDecoderLayer):
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
        attention_scale: float | None,
        attention_logit_soft_cap: float | None,
        sliding_window_size: int | None,
        eps: float,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        attention_key, mlp_key = jax.random.split(key)

        self.attention_norm = attention_norm_config(model_dim, eps)
        self.attention = attention_config(
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            scale=attention_scale,
            logit_soft_cap=attention_logit_soft_cap,
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
        x, kv_cache = self.attention(x, positional_embeddings, kv_cache, mask, return_updated_kv_cache)
        x = residual + x

        residual = x
        x = vmap(self.mlp_norm, in_axes=0)(x)
        x = residual + vmap(self.mlp, in_axes=0)(x)

        return DecoderLayerOutput(output=x, kv_cache=kv_cache)

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            attention_norm=self.attention_norm.export_weights(),
            attention=self.attention.export_weights(),
            mlp_norm=self.mlp_norm.export_weights(),
            mlp=self.mlp.export_weights(),
        )


@dataclass
class PreNormDecoderLayerConfig[
    MLPNormType: AbstractNormalization,
    MLPType: AbstractMLP,
    AttentionNormType: AbstractNormalization,
    AttentionType: AttentionBase,
](AbstractDecoderLayerConfig[PreNormDecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]]):
    attention_norm_config: NormalizationConfigType
    attention_config: AttentionConfigType
    mlp_norm_config: NormalizationConfigType
    mlp_config: MLPConfigType

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
        attention_scale: float | None,
        attention_logit_soft_cap: float | None,
        sliding_window_size: int | None,
        eps: float,
        key: PRNGKeyArray,
    ) -> PreNormDecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]:
        return PreNormDecoderLayer(
            attention_norm_config=self.attention_norm_config,  # type: ignore
            attention_config=self.attention_config,  # type: ignore
            mlp_norm_config=self.mlp_norm_config,  # type: ignore
            mlp_config=self.mlp_config,  # type: ignore
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            activation=activation,
            use_mlp_bias=use_mlp_bias,
            use_attention_qkv_bias=use_attention_qkv_bias,
            use_attention_out_bias=use_attention_out_bias,
            attention_scale=attention_scale,
            attention_logit_soft_cap=attention_logit_soft_cap,
            sliding_window_size=sliding_window_size,
            eps=eps,
            key=key,
        )


class PrePostNormDecoderLayer[
    MLPPreNormType: AbstractNormalization,
    MLPType: AbstractMLP,
    MLPPostNormType: AbstractNormalization,
    AttentionPreNormType: AbstractNormalization,
    AttentionType: AttentionBase,
    AttentionPostNormType: AbstractNormalization,
](AbstractDecoderLayer):
    attention_pre_norm: AttentionPreNormType
    attention: AttentionType
    attention_post_norm: AttentionPostNormType
    mlp_pre_norm: MLPPreNormType
    mlp: MLPType
    mlp_post_norm: MLPPostNormType

    def __init__(
        self,
        *,
        attention_pre_norm_config: AbstractNormalizationConfig[AttentionPreNormType],
        attention_config: AbstractAttentionConfig[AttentionType],
        attention_post_norm_config: AbstractNormalizationConfig[AttentionPostNormType],
        mlp_pre_norm_config: AbstractNormalizationConfig[MLPPreNormType],
        mlp_config: AbstractMLPConfig[MLPType],
        mlp_post_norm_config: AbstractNormalizationConfig[MLPPostNormType],
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        activation: Activation,
        use_mlp_bias: bool,
        use_attention_qkv_bias: bool,
        use_attention_out_bias: bool,
        attention_scale: float | None,
        attention_logit_soft_cap: float | None,
        sliding_window_size: int | None,
        eps: float,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        attention_key, mlp_key = jax.random.split(key)

        self.attention_pre_norm = attention_pre_norm_config(model_dim, eps)
        self.attention = attention_config(
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            scale=attention_scale,
            logit_soft_cap=attention_logit_soft_cap,
            sliding_window_size=sliding_window_size,
            use_qkv_bias=use_attention_qkv_bias,
            use_out_bias=use_attention_out_bias,
            key=attention_key,
        )
        self.attention_post_norm = attention_post_norm_config(model_dim, eps)

        self.mlp_pre_norm = mlp_pre_norm_config(model_dim, eps)
        self.mlp = mlp_config(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            use_bias=use_mlp_bias,
            activation=activation,
            key=mlp_key,
        )
        self.mlp_post_norm = mlp_post_norm_config(model_dim, eps)

    def __call__(
        self,
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> DecoderLayerOutput:
        residual = x
        x = vmap(self.attention_pre_norm, in_axes=0)(x)
        x, kv_cache = self.attention(x, positional_embeddings, kv_cache, mask, return_updated_kv_cache)
        x = vmap(self.attention_post_norm, in_axes=0)(x)
        x = residual + x

        residual = x
        x = vmap(self.mlp_pre_norm, in_axes=0)(x)
        x = vmap(self.mlp, in_axes=0)(x)
        x = vmap(self.mlp_post_norm, in_axes=0)(x)
        x = residual + x

        return DecoderLayerOutput(output=x, kv_cache=kv_cache)

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            attention_pre_norm=self.attention_pre_norm.export_weights(),
            attention=self.attention.export_weights(),
            attention_post_norm=self.attention_post_norm.export_weights(),
            mlp_pre_norm=self.mlp_pre_norm.export_weights(),
            mlp=self.mlp.export_weights(),
            mlp_post_norm=self.mlp_post_norm.export_weights(),
        )


@dataclass
class PrePostNormDecoderLayerConfig[
    MLPPreNormType: AbstractNormalization,
    MLPType: AbstractMLP,
    MLPPostNormType: AbstractNormalization,
    AttentionPreNormType: AbstractNormalization,
    AttentionType: AttentionBase,
    AttentionPostNormType: AbstractNormalization,
](
    AbstractDecoderLayerConfig[
        PrePostNormDecoderLayer[
            MLPPreNormType,
            MLPType,
            MLPPostNormType,
            AttentionPreNormType,
            AttentionType,
            AttentionPostNormType,
        ]
    ],
):
    attention_pre_norm_config: NormalizationConfigType
    attention_config: AttentionConfigType
    attention_post_norm_config: NormalizationConfigType
    mlp_pre_norm_config: NormalizationConfigType
    mlp_config: MLPConfigType
    mlp_post_norm_config: NormalizationConfigType

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
        attention_scale: float | None,
        attention_logit_soft_cap: float | None,
        sliding_window_size: int | None,
        eps: float,
        key: PRNGKeyArray,
    ) -> PrePostNormDecoderLayer[
        MLPPreNormType,
        MLPType,
        MLPPostNormType,
        AttentionPreNormType,
        AttentionType,
        AttentionPostNormType,
    ]:
        return PrePostNormDecoderLayer(
            attention_pre_norm_config=self.attention_pre_norm_config,  # type: ignore
            attention_config=self.attention_config,  # type: ignore
            attention_post_norm_config=self.attention_post_norm_config,  # type: ignore
            mlp_pre_norm_config=self.mlp_pre_norm_config,  # type: ignore
            mlp_config=self.mlp_config,  # type: ignore
            mlp_post_norm_config=self.mlp_post_norm_config,  # type: ignore
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            activation=activation,
            use_mlp_bias=use_mlp_bias,
            use_attention_qkv_bias=use_attention_qkv_bias,
            use_attention_out_bias=use_attention_out_bias,
            attention_scale=attention_scale,
            attention_logit_soft_cap=attention_logit_soft_cap,
            sliding_window_size=sliding_window_size,
            eps=eps,
            key=key,
        )


DecoderLayerConfigType = PreNormDecoderLayerConfig | PrePostNormDecoderLayerConfig

register_config_union(DecoderLayerConfigType)
