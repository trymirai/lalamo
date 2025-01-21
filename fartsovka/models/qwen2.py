from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.modules.attention import Attention, AttentionFactory
from fartsovka.modules.decoder import Decoder, DecoderFactory
from fartsovka.modules.decoder_layer import DecoderLayer, DecoderLayerFactory
from fartsovka.modules.embedding import Embedding, EmbeddingFactory
from fartsovka.modules.linear import Linear, LinearFactory
from fartsovka.modules.mlp import MLP, MLPFactory
from fartsovka.modules.normalization import RMSNorm, RMSNormFactory
from fartsovka.modules.rope import RoPEBase, RoPEFactory

__all__ = [
    "Qwen2MLP",
    "Qwen2Attention",
    "Qwen2DecoderLayer",
    "Qwen2",
    "get_qwen2",
    "get_qwen2_factory",
]

type Qwen2MLP = MLP[Linear]

type Qwen2Attention = Attention[Linear, Linear]

type Qwen2DecoderLayer = DecoderLayer[
    RMSNorm,
    Qwen2MLP,
    RMSNorm,
    Qwen2Attention,
]

type Qwen2 = Decoder[
    Embedding,
    RMSNorm,
    Qwen2MLP,
    RMSNorm,
    Qwen2Attention,
    RoPEBase,
]


def get_qwen2_factory(
    precision: DType,
    accumulation_precision: DType,
    rope_scaling_factor: float,
    rope_beta_fast: float,
    rope_beta_slow: float,
) -> DecoderFactory[
    Embedding,
    RMSNorm,
    Qwen2MLP,
    RMSNorm,
    Qwen2Attention,
    RoPEBase,
]:
    return DecoderFactory(
        embedding_factory=EmbeddingFactory(precision=precision),
        rope_factory=RoPEFactory(
            precision=precision,
            # scaling_factor=rope_scaling_factor,
            # beta_fast=rope_beta_fast,
            # beta_slow=rope_beta_slow,
        ),
        layer_factory=DecoderLayerFactory(
            attention_norm_factory=RMSNormFactory(
                precision=precision,
                accumulation_precision=accumulation_precision,
            ),
            attention_factory=AttentionFactory(
                qkv_projection_factory=LinearFactory(precision=precision),
                out_projection_factory=LinearFactory(precision=precision),
            ),
            mlp_norm_factory=RMSNormFactory(
                precision=precision,
                accumulation_precision=accumulation_precision,
            ),
            mlp_factory=MLPFactory(
                linear_factory=LinearFactory(precision=precision),
            ),
        ),
        output_norm_factory=RMSNormFactory(
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
) -> Qwen2:
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
