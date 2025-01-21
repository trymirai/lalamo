from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .attention import AttentionBase, AttentionFactoryBase
from .common import FartsovkaModule, ParameterDict
from .kv_cache import KVCacheLayerSlice
from .mlp import MLPBase, MLPFactoryBase
from .normalization import NormalizationBase, NormalizationFactoryBase
from .rope import PositionalEmbeddings

__all__ = ["DecoderLayer", "DecoderLayerFactory"]


class DecoderLayerOutput(NamedTuple):
    output: Float[Array, "suffix_tokens channels"]
    kv_cache: KVCacheLayerSlice | None


class DecoderLayer[
    MLPNormType: NormalizationBase,
    MLPType: MLPBase,
    AttentionNormType: NormalizationBase,
    AttentionType: AttentionBase,
](FartsovkaModule):
    model_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    use_attention_qkv_bias: bool = eqx.field(static=True)
    use_attention_out_bias: bool = eqx.field(static=True)
    use_mlp_bias: bool = eqx.field(static=True)

    attention_norm: AttentionNormType
    attention: AttentionType
    mlp_norm: MLPNormType
    mlp: MLPType

    def __init__(
        self,
        *,
        attention_norm_factory: NormalizationFactoryBase[AttentionNormType],
        attention_factory: AttentionFactoryBase[AttentionType],
        mlp_norm_factory: NormalizationFactoryBase[MLPNormType],
        mlp_factory: MLPFactoryBase[MLPType],
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        use_attention_qkv_bias: bool,
        use_attention_out_bias: bool,
        use_mlp_bias: bool,
        eps: float,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.use_attention_qkv_bias = use_attention_qkv_bias
        self.use_attention_out_bias = use_attention_out_bias
        self.use_mlp_bias = use_mlp_bias

        attention_key, mlp_key = jax.random.split(key)

        self.attention_norm = attention_norm_factory(model_dim, eps)
        self.attention = attention_factory(
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            use_qkv_bias=use_attention_qkv_bias,
            use_out_bias=use_attention_out_bias,
            key=attention_key,
        )
        self.mlp_norm = mlp_norm_factory(model_dim, eps)
        self.mlp = mlp_factory(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            use_bias=use_mlp_bias,
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
class DecoderLayerFactory[
    MLPNormType: NormalizationBase,
    MLPType: MLPBase,
    AttentionNormType: NormalizationBase,
    AttentionType: AttentionBase,
]:
    attention_norm_factory: NormalizationFactoryBase[AttentionNormType]
    attention_factory: AttentionFactoryBase[AttentionType]
    mlp_norm_factory: NormalizationFactoryBase[MLPNormType]
    mlp_factory: MLPFactoryBase[MLPType]

    def __call__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        use_attention_qkv_bias: bool,
        use_attention_out_bias: bool,
        use_mlp_bias: bool,
        eps: float,
        key: PRNGKeyArray,
    ) -> DecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]:
        return DecoderLayer(
            attention_norm_factory=self.attention_norm_factory,
            attention_factory=self.attention_factory,
            mlp_norm_factory=self.mlp_norm_factory,
            mlp_factory=self.mlp_factory,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            use_attention_qkv_bias=use_attention_qkv_bias,
            use_attention_out_bias=use_attention_out_bias,
            use_mlp_bias=use_mlp_bias,
            eps=eps,
            key=key,
        )
