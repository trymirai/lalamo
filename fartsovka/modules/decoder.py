from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .attention import AbstractAttention
from .common import FartsovkaModule, ModuleConfig, ParameterDict
from .decoder_layer import DecoderLayer, DecoderLayerConfig
from .embedding import AbstractEmbedding, AbstractEmbeddingConfig
from .kv_cache import KVCacheLayerSlice
from .mlp import AbstractMLP
from .normalization import AbstractNormalization, RMSNorm, RMSNormConfig
from .rope import AbstractRoPE, AbstractRoPEConfig

__all__ = ["Decoder", "DecoderConfig"]


class DecoderOutput(NamedTuple):
    output: Float[Array, "suffix_tokens channels"]
    kv_cache: list[KVCacheLayerSlice] | None = None


class Decoder[
    EmbeddingType: AbstractEmbedding,
    MLPNormType: AbstractNormalization,
    MLPType: AbstractMLP,
    AttentionNormType: AbstractNormalization,
    AttentionType: AbstractAttention,
    RoPEType: AbstractRoPE,
](FartsovkaModule):
    num_layers: int = eqx.field(static=True)
    vocab_dim: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    use_attention_qkv_bias: bool = eqx.field(static=True)
    use_attention_out_bias: bool = eqx.field(static=True)
    use_mlp_bias: bool = eqx.field(static=True)
    rope_theta: float = eqx.field(static=True)
    max_sequence_length: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    embedding: EmbeddingType
    rope: AbstractRoPE
    layers: list[DecoderLayer[MLPNormType, MLPType, AttentionNormType, AttentionType]]
    output_norm: RMSNorm

    def __init__(
        self,
        *,
        num_layers: int,
        embedding_config: AbstractEmbeddingConfig[EmbeddingType],
        rope_config: AbstractRoPEConfig[RoPEType],
        layer_config: DecoderLayerConfig[MLPNormType, MLPType, AttentionNormType, AttentionType],
        output_norm_config: RMSNormConfig,
        vocab_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        use_attention_qkv_bias: bool,
        use_attention_out_bias: bool,
        use_mlp_bias: bool,
        rope_theta: float,
        max_sequence_length: int,
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
        self.use_attention_qkv_bias = use_attention_qkv_bias
        self.use_attention_out_bias = use_attention_out_bias
        self.use_mlp_bias = use_mlp_bias
        self.rope_theta = rope_theta
        self.max_sequence_length = max_sequence_length
        self.eps = eps

        embedding_key, layers_key = jax.random.split(key)
        layer_keys = jax.random.split(layers_key, num_layers)

        self.embedding = embedding_config(vocab_dim, model_dim, key=embedding_key)
        self.rope = rope_config(head_dim, max_sequence_length, theta=rope_theta)
        self.layers = [
            layer_config(
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                head_dim=head_dim,
                use_attention_qkv_bias=use_attention_qkv_bias,
                use_attention_out_bias=use_attention_out_bias,
                use_mlp_bias=use_mlp_bias,
                eps=eps,
                key=layer_key,
            )
            for layer_key in layer_keys
        ]
        self.output_norm = output_norm_config(model_dim, eps)

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

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            embedding=self.embedding.export_weights(),
            rope=self.rope.export_weights(),
            layers=[layer.export_weights() for layer in self.layers],
            output_norm=self.output_norm.export_weights(),
        )


@dataclass
class DecoderConfig[
    EmbeddingType: AbstractEmbedding,
    MLPNormType: AbstractNormalization,
    MLPType: AbstractMLP,
    AttentionNormType: AbstractNormalization,
    AttentionType: AbstractAttention,
    RoPEType: AbstractRoPE,
](ModuleConfig[Decoder[EmbeddingType, MLPNormType, MLPType, AttentionNormType, AttentionType, RoPEType]]):
    embedding_config: AbstractEmbeddingConfig[EmbeddingType]
    rope_config: AbstractRoPEConfig[RoPEType]
    layer_config: DecoderLayerConfig[MLPNormType, MLPType, AttentionNormType, AttentionType]
    output_norm_config: RMSNormConfig

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
        use_attention_qkv_bias: bool,
        use_attention_out_bias: bool,
        use_mlp_bias: bool,
        rope_theta: float,
        max_sequence_length: int,
        eps: float,
        key: PRNGKeyArray,
    ) -> Decoder[EmbeddingType, MLPNormType, MLPType, AttentionNormType, AttentionType, RoPEType]:
        return Decoder(
            num_layers=num_layers,
            embedding_config=self.embedding_config,
            rope_config=self.rope_config,
            layer_config=self.layer_config,
            output_norm_config=self.output_norm_config,
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            use_attention_qkv_bias=use_attention_qkv_bias,
            use_attention_out_bias=use_attention_out_bias,
            use_mlp_bias=use_mlp_bias,
            rope_theta=rope_theta,
            max_sequence_length=max_sequence_length,
            eps=eps,
            key=key,
        )
