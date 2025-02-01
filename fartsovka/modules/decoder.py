from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .activations import Activation
from .common import DummyUnionMember, FartsovkaModule, ParameterDict, register_config_union
from .decoder_layer import AbstractDecoderLayer, AbstractDecoderLayerConfig, DecoderLayerConfigType
from .embedding import AbstractEmbedding, AbstractEmbeddingConfig, EmbeddingConfigType
from .kv_cache import KVCacheLayerSlice
from .normalization import RMSNorm, RMSNormConfig
from .rope import AbstractRoPE, AbstractRoPEConfig, RoPEConfigType

__all__ = [
    "Decoder",
    "DecoderConfig",
    "DecoderConfigType",
    "DecoderOutput",
]


class DecoderOutput(NamedTuple):
    output: Float[Array, "suffix_tokens channels"]
    kv_cache: list[KVCacheLayerSlice] | None = None


class Decoder[
    EmbeddingType: AbstractEmbedding,
    DecoderLayerType: AbstractDecoderLayer,
    RoPEType: AbstractRoPE,
](FartsovkaModule):
    embedding: EmbeddingType
    rope: AbstractRoPE
    layers: list[DecoderLayerType]
    output_norm: RMSNorm

    def __init__(
        self,
        *,
        num_layers: int,
        embedding_config: AbstractEmbeddingConfig[EmbeddingType],
        rope_config: AbstractRoPEConfig[RoPEType],
        layer_config: AbstractDecoderLayerConfig[DecoderLayerType],
        output_norm_config: RMSNormConfig,
        vocab_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        embedding_input_scale: float | None,
        output_logits_soft_cap: float | None,
        activation: Activation,
        use_mlp_bias: bool,
        use_attention_qkv_bias: bool,
        use_attention_out_bias: bool,
        attention_scale: float | None,
        attention_logit_soft_cap: float | None,
        sliding_window_sizes: list[int | None] | None,
        rope_theta: float,
        max_sequence_length: int,
        eps: float,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()
        not_none_sliding_window_sizes = sliding_window_sizes or [None] * num_layers

        embedding_key, layers_key = jax.random.split(key)
        layer_keys = jax.random.split(layers_key, num_layers)

        self.embedding = embedding_config(
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            input_scale=embedding_input_scale,
            logits_soft_cap=output_logits_soft_cap,
            key=embedding_key,
        )
        self.rope = rope_config(head_dim, max_sequence_length, theta=rope_theta)
        self.layers = [
            layer_config(
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                head_dim=head_dim,
                activation=activation,
                use_attention_qkv_bias=use_attention_qkv_bias,
                use_attention_out_bias=use_attention_out_bias,
                use_mlp_bias=use_mlp_bias,
                attention_scale=attention_scale,
                attention_logit_soft_cap=attention_logit_soft_cap,
                sliding_window_size=sliding_window_size,
                eps=eps,
                key=layer_key,
            )
            for layer_key, sliding_window_size in zip(layer_keys, not_none_sliding_window_sizes, strict=True)
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
    DecoderLayerType: AbstractDecoderLayer,
    RoPEType: AbstractRoPE,
]:
    embedding_config: EmbeddingConfigType
    rope_config: RoPEConfigType
    layer_config: DecoderLayerConfigType
    output_norm_config: RMSNormConfig

    num_layers: int
    vocab_dim: int
    model_dim: int
    hidden_dim: int
    num_heads: int
    num_groups: int
    head_dim: int
    embedding_input_scale: float | None
    output_logits_soft_cap: float | None
    activation: Activation
    use_mlp_bias: bool
    use_attention_qkv_bias: bool
    use_attention_out_bias: bool
    attention_scale: float | None
    attention_logit_soft_cap: float | None
    sliding_window_sizes: list[int | None] | None
    rope_theta: float
    max_sequence_length: int
    eps: float

    def __call__(
        self,
        *,
        key: PRNGKeyArray,
    ) -> Decoder[EmbeddingType, DecoderLayerType, RoPEType]:
        return Decoder(
            num_layers=self.num_layers,
            embedding_config=self.embedding_config,  # type: ignore
            rope_config=self.rope_config,
            layer_config=self.layer_config,  # type: ignore
            output_norm_config=self.output_norm_config,
            vocab_dim=self.vocab_dim,
            model_dim=self.model_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            embedding_input_scale=self.embedding_input_scale,
            output_logits_soft_cap=self.output_logits_soft_cap,
            activation=self.activation,
            use_mlp_bias=self.use_mlp_bias,
            use_attention_qkv_bias=self.use_attention_qkv_bias,
            use_attention_out_bias=self.use_attention_out_bias,
            attention_scale=self.attention_scale,
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            sliding_window_sizes=self.sliding_window_sizes,
            rope_theta=self.rope_theta,
            max_sequence_length=self.max_sequence_length,
            eps=self.eps,
            key=key,
        )


DecoderConfigType = DecoderConfig | DummyUnionMember

register_config_union(DecoderConfigType)
