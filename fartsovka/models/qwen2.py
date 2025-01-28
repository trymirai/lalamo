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
from fartsovka.modules.rope import AbstractRoPE, RoPEConfig

__all__ = [
    "get_qwen2_factory",
    "get_qwen2",
    "Qwen2Decoder",
    "Qwen2Attention",
    "Qwen2DecoderLayer",
    "Qwen2MLP",
]

type Qwen2MLP = MLP[Linear]

type Qwen2Attention = Attention[Linear, Linear]

type Qwen2DecoderLayer = DecoderLayer[
    RMSNorm,
    Qwen2MLP,
    RMSNorm,
    Qwen2Attention,
]

type Qwen2Decoder = Decoder[
    Embedding,
    RMSNorm,
    Qwen2MLP,
    RMSNorm,
    Qwen2Attention,
    AbstractRoPE,
]


def get_qwen2_factory(
    precision: DType,
    accumulation_precision: DType,
    rope_scaling_factor: float,
    rope_beta_fast: float,
    rope_beta_slow: float,
) -> DecoderConfig[
    Embedding,
    RMSNorm,
    Qwen2MLP,
    RMSNorm,
    Qwen2Attention,
    AbstractRoPE,
]:
    return DecoderConfig(
        embedding_config=EmbeddingConfig(precision=precision),
        rope_config=RoPEConfig(
            precision=precision,
            # scaling_factor=rope_scaling_factor,
            # beta_fast=rope_beta_fast,
            # beta_slow=rope_beta_slow,
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


def get_qwen2(
    *,
    num_layers: int,
    vocab_dim: int,
    model_dim: int,
    hidden_dim: int,
    num_heads: int,
    num_groups: int,
    head_dim: int,
    rope_theta: float,
    rope_scaling_factor: float,
    rope_beta_fast: float,
    rope_beta_slow: float,
    max_sequence_length: int,
    eps: float,
    key: PRNGKeyArray,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> Qwen2Decoder:
    factory = get_qwen2_factory(
        precision=precision,
        accumulation_precision=accumulation_precision,
        rope_scaling_factor=rope_scaling_factor,
        rope_beta_fast=rope_beta_fast,
        rope_beta_slow=rope_beta_slow,
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
        use_attention_qkv_bias=True,
        use_attention_out_bias=False,
        use_mlp_bias=False,
        eps=eps,
        key=key,
    )
