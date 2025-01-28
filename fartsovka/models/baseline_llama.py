from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.modules.attention import Attention, AttentionConfig
from fartsovka.modules.decoder import Decoder, DecoderConfig
from fartsovka.modules.decoder_layer import DecoderLayer, DecoderLayerConfig
from fartsovka.modules.embedding import Embedding, EmbeddingConfig
from fartsovka.modules.linear import Linear, LinearConfig
from fartsovka.modules.mlp import MLP, MLPConfig
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.rope import LlamaRoPE, LlamaRoPEConfig

__all__ = [
    "get_baseline_llama_config",
    "get_baseline_llama",
    "LlamaAttention",
    "LlamaDecoder",
    "LlamaDecoderLayer",
    "LlamaMLP",
]

type LlamaMLP = MLP[Linear]

type LlamaAttention = Attention[Linear, Linear]

type LlamaDecoderLayer = DecoderLayer[
    RMSNorm,
    LlamaMLP,
    RMSNorm,
    LlamaAttention,
]

type LlamaDecoder = Decoder[
    Embedding,
    RMSNorm,
    LlamaMLP,
    RMSNorm,
    LlamaAttention,
    LlamaRoPE,
]


def get_baseline_llama_config(
    precision: DType,
    accumulation_precision: DType,
    original_context_length: int,
    rope_scaling_factor: float,
    rope_low_frequency_factor: float,
    rope_high_frequency_factor: float,
) -> DecoderConfig[
    Embedding,
    RMSNorm,
    LlamaMLP,
    RMSNorm,
    LlamaAttention,
    LlamaRoPE,
]:
    return DecoderConfig(
        embedding_config=EmbeddingConfig(precision=precision),
        rope_config=LlamaRoPEConfig(
            precision=precision,
            original_context_length=original_context_length,
            scaling_factor=rope_scaling_factor,
            low_frequency_factor=rope_low_frequency_factor,
            high_frequency_factor=rope_high_frequency_factor,
        ),
        layer_config=DecoderLayerConfig(
            attention_norm_config=RMSNormConfig(
                precision=precision,
                accumulation_precision=accumulation_precision,
            ),
            attention_config=AttentionConfig(
                qkv_projection_config=LinearConfig(precision=precision),
                out_projection_config=LinearConfig(precision=precision),
            ),
            mlp_norm_config=RMSNormConfig(
                precision=precision,
                accumulation_precision=accumulation_precision,
            ),
            mlp_config=MLPConfig(
                linear_config=LinearConfig(precision=precision),
            ),
        ),
        output_norm_config=RMSNormConfig(
            precision=precision,
            accumulation_precision=accumulation_precision,
        ),
    )


def get_baseline_llama(
    *,
    num_layers: int,
    vocab_dim: int,
    model_dim: int,
    hidden_dim: int,
    num_heads: int,
    num_groups: int,
    head_dim: int,
    max_sequence_length: int,
    rope_theta: float,
    rope_scaling_factor: float,
    rope_original_context_length: int,
    rope_low_frequency_factor: float,
    rope_high_frequency_factor: float,
    eps: float,
    key: PRNGKeyArray,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> LlamaDecoder:
    factory = get_baseline_llama_config(
        precision=precision,
        accumulation_precision=accumulation_precision,
        original_context_length=rope_original_context_length,
        rope_scaling_factor=rope_scaling_factor,
        rope_low_frequency_factor=rope_low_frequency_factor,
        rope_high_frequency_factor=rope_high_frequency_factor,
    )
    return factory(
        num_layers=num_layers,
        vocab_dim=vocab_dim,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_groups=num_groups,
        head_dim=head_dim,
        max_sequence_length=max_sequence_length,
        rope_theta=rope_theta,
        use_attention_qkv_bias=False,
        use_attention_out_bias=False,
        use_mlp_bias=False,
        eps=eps,
        key=key,
    )
