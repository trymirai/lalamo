from dataclasses import dataclass

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .attention import AttentionBase
from .decoder_layer import DecoderLayer, DecoderLayerFactory
from .embedding import Embedding, EmbeddingFactory
from .kv_cache import KVCacheLayerSlice
from .mlp import MLPBase
from .normalisation import NormalisationBase, RMSNorm, RMSNormFactory
from .rope import RoPE, RoPEFactory

__all__ = ["DecoderLayer", "DecoderLayerFactory"]


class Decoder[
    MLPNormType: NormalisationBase,
    MLPType: MLPBase,
    AttentionNormType: NormalisationBase,
    AttentionType: AttentionBase,
](eqx.Module):
    num_layers: int = eqx.field(static=True)
    vocab_dim: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    max_sequence_length: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    embedding: Embedding
    rope: RoPE
    layers: list[DecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]]
    out_norm: RMSNorm

    def __init__(
        self,
        num_layers: int,
        embedding_factory: EmbeddingFactory,
        rope_factory: RoPEFactory,
        layer_factory: DecoderLayerFactory[MLPNormType, MLPType, AttentionNormType, AttentionType],
        out_norm_factory: RMSNormFactory,
        vocab_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        eps: float,
        max_sequence_length: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.num_layers = num_layers
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.eps = eps

        embedding_key, layers_key = jax.random.split(key)
        layer_keys = jax.random.split(layers_key, num_layers)

        self.embedding = embedding_factory(vocab_dim, model_dim, key=embedding_key)
        self.rope = rope_factory(head_dim, max_sequence_length)
        self.layers = [
            layer_factory(model_dim, hidden_dim, num_heads, num_groups, head_dim, eps, key=layer_key)
            for layer_key in layer_keys
        ]
        self.out_norm = out_norm_factory(model_dim, eps)

    def __call__(
        self,
        token_ids: Int[Array, " suffix_tokens"],
        token_positions: Int[Array, " suffix_tokens"],
        kv_cache: list[KVCacheLayerSlice] | None = None,
        mask: Bool[Array, "suffix_tokens prefix_tokens+suffix_tokens"] | None = None,
    ) -> Float[Array, "suffix_tokens token_ids"]:
        maybe_kv_cache = kv_cache or ([None] * len(self.layers))
        x = self.embedding.embed(token_ids)
        positional_embeddings = self.rope(token_positions)
        for layer, kv_cache_slice in zip(self.layers, maybe_kv_cache, strict=True):
            x = layer(x, positional_embeddings, kv_cache_slice, mask)
        x = self.out_norm(x)
        return vmap(self.embedding.readout, in_axes=0)(x)


@dataclass
class DecoderFactory[
    MLPNormType: NormalisationBase,
    MLPType: MLPBase,
    AttentionNormType: NormalisationBase,
    AttentionType: AttentionBase,
]:
    embedding_factory: EmbeddingFactory
    rope_factory: RoPEFactory
    layer_factory: DecoderLayerFactory[MLPNormType, MLPType, AttentionNormType, AttentionType]
    out_norm_factory: RMSNormFactory

    def __call__(
        self,
        num_layers: int,
        vocab_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        eps: float,
        max_sequence_length: int,
        *,
        key: PRNGKeyArray,
    ) -> Decoder[MLPNormType, MLPType, AttentionNormType, AttentionType]:
        return Decoder(
            num_layers,
            self.embedding_factory,
            self.rope_factory,
            self.layer_factory,
            self.out_norm_factory,
            vocab_dim,
            model_dim,
            hidden_dim,
            num_heads,
            num_groups,
            head_dim,
            eps,
            max_sequence_length,
            key=key,
        )
