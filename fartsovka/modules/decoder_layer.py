from dataclasses import dataclass

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .attention import AttentionBase, AttentionFactoryBase
from .kv_cache import KVCacheLayerSlice
from .mlp import MLPBase, MLPFactoryBase
from .normalisation import NormalisationBase, NormalisationFactory
from .rope import PositionalEmbeddings

__all__ = ["DecoderLayer", "DecoderLayerFactory"]


class DecoderLayer[
    MLPNormType: NormalisationBase,
    MLPType: MLPBase,
    AttentionNormType: NormalisationBase,
    AttentionType: AttentionBase,
](eqx.Module):
    model_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    attention_norm: AttentionNormType
    attention: AttentionType
    mlp_norm: MLPNormType
    mlp: MLPType

    def __init__(
        self,
        attention_norm_factory: NormalisationFactory[AttentionNormType],
        attention_factory: AttentionFactoryBase[AttentionType],
        mlp_norm_factory: NormalisationFactory[MLPNormType],
        mlp_factory: MLPFactoryBase[MLPType],
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        eps: float,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim

        attention_key, mlp_key = jax.random.split(key)

        self.attention_norm = attention_norm_factory(model_dim, eps)
        self.attention = attention_factory(model_dim, num_heads, num_groups, head_dim, key=attention_key)
        self.mlp_norm = mlp_norm_factory(model_dim, eps)
        self.mlp = mlp_factory(model_dim, hidden_dim, key=mlp_key)

    def __call__(
        self,
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice,
        mask: Bool[Array, "suffix_tokens prefix_tokens+suffix_tokens"],
    ) -> Float[Array, "suffix_tokens channels"]:
        residual = x
        x = vmap(self.attention_norm, in_axes=0)(x)
        x = self.attention(x, positional_embeddings, kv_cache, mask)
        x = x + residual

        residual = x
        x = vmap(self.mlp_norm, in_axes=0)(x)
        x = self.mlp(x)
        x = x + residual

        return x


@dataclass
class DecoderLayerFactory[
    MLPNormType: NormalisationBase,
    MLPType: MLPBase,
    AttentionNormType: NormalisationBase,
    AttentionType: AttentionBase,
]:
    pre_attention_norm_factory: NormalisationFactory[AttentionNormType]
    attention_factory: AttentionFactoryBase[AttentionType]
    pre_mlp_norm_factory: NormalisationFactory[MLPNormType]
    mlp_factory: MLPFactoryBase[MLPType]

    def __call__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        eps: float,
        *,
        key: PRNGKeyArray,
    ) -> DecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]:
        return DecoderLayer(
            self.pre_attention_norm_factory,
            self.attention_factory,
            self.pre_mlp_norm_factory,
            self.mlp_factory,
            model_dim,
            hidden_dim,
            num_heads,
            num_groups,
            head_dim,
            eps=eps,
            key=key,
        )
