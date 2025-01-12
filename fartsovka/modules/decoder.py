from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .attention import AttentionBase
from .decoder_layer import DecoderLayer, DecoderLayerFactory
from .embedding import EmbeddingBase, EmbeddingFactoryBase
from .kv_cache import KVCacheLayerSlice
from .mlp import MLPBase
from .normalization import NormalizationBase, RMSNorm, RMSNormFactory
from .rope import RoPE, RoPEFactory, RoPEParams

__all__ = ["Decoder", "DecoderFactory"]


class DecoderOutput(NamedTuple):
    output: Float[Array, "suffix_tokens channels"]
    kv_cache: list[KVCacheLayerSlice] | None = None


class Decoder[
    EmbeddingType: EmbeddingBase,
    MLPNormType: NormalizationBase,
    MLPType: MLPBase,
    AttentionNormType: NormalizationBase,
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
    rope_params: RoPEParams = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    embedding: EmbeddingType
    rope: RoPE
    layers: list[DecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]]
    output_norm: RMSNorm

    def __init__(
        self,
        *,
        num_layers: int,
        embedding_factory: EmbeddingFactoryBase[EmbeddingType],
        rope_factory: RoPEFactory,
        layer_factory: DecoderLayerFactory[MLPNormType, MLPType, AttentionNormType, AttentionType],
        output_norm_factory: RMSNormFactory,
        vocab_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        max_sequence_length: int,
        rope_params: RoPEParams,
        eps: float,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.rope_params = rope_params
        self.eps = eps

        embedding_key, layers_key = jax.random.split(key)
        layer_keys = jax.random.split(layers_key, num_layers)

        self.embedding = embedding_factory(vocab_dim, model_dim, key=embedding_key)
        self.rope = rope_factory(head_dim, max_sequence_length, params=rope_params)
        self.layers = [
            layer_factory(
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                head_dim=head_dim,
                eps=eps,
                key=layer_key,
            )
            for layer_key in layer_keys
        ]
        self.output_norm = output_norm_factory(model_dim, eps)

    def __call__(
        self,
        token_ids: Int[Array, " suffix_tokens"],
        token_positions: Int[Array, " suffix_tokens"],
        kv_cache: list[KVCacheLayerSlice] | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> DecoderOutput:
        maybe_kv_cache = kv_cache or ([None] * len(self.layers))
        x = self.embedding.embed(token_ids)
        positional_embeddings = self.rope(token_positions)
        updated_kv_cache = []
        for layer, kv_cache_slice in zip(self.layers, maybe_kv_cache, strict=True):
            decoder_layer_output = layer(
                x,
                positional_embeddings,
                kv_cache_slice,
                mask,
                return_updated_kv_cache,
            )
            x = decoder_layer_output.output
            updated_kv_cache.append(decoder_layer_output.kv_cache)
        x = vmap(self.output_norm, in_axes=0)(x)
        result = vmap(self.embedding.readout, in_axes=0)(x)
        return DecoderOutput(output=result, kv_cache=updated_kv_cache or None)


@dataclass
class DecoderFactory[
    EmbeddingType: EmbeddingBase,
    MLPNormType: NormalizationBase,
    MLPType: MLPBase,
    AttentionNormType: NormalizationBase,
    AttentionType: AttentionBase,
]:
    embedding_factory: EmbeddingFactoryBase[EmbeddingType]
    rope_factory: RoPEFactory
    layer_factory: DecoderLayerFactory[MLPNormType, MLPType, AttentionNormType, AttentionType]
    output_norm_factory: RMSNormFactory

    def __call__(
        self,
        *,
        num_layers: int,
        vocab_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        max_sequence_length: int,
        rope_params: RoPEParams,
        eps: float,
        key: PRNGKeyArray,
    ) -> Decoder[EmbeddingType, MLPNormType, MLPType, AttentionNormType, AttentionType]:
        return Decoder(
            num_layers=num_layers,
            embedding_factory=self.embedding_factory,
            rope_factory=self.rope_factory,
            layer_factory=self.layer_factory,
            output_norm_factory=self.output_norm_factory,
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            max_sequence_length=max_sequence_length,
            rope_params=rope_params,
            eps=eps,
            key=key,
        )
