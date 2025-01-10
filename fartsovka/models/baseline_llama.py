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
from fartsovka.modules.rope import RoPEFactory, RoPEParams

__all__ = [
    "BaselineMLP",
    "BaselineAttention",
    "BaselineDecoderLayer",
    "BaselineLlama",
    "get_baseline_llama",
    "get_baseline_llama_factory",
]

type BaselineMLP = MLP[Linear]

type BaselineAttention = Attention[Linear, Linear]

type BaselineDecoderLayer = DecoderLayer[
    RMSNorm,
    BaselineMLP,
    RMSNorm,
    BaselineAttention,
]

type BaselineLlama = Decoder[
    Embedding,
    RMSNorm,
    BaselineMLP,
    RMSNorm,
    BaselineAttention,
]


def get_baseline_llama_factory(
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> DecoderFactory[
    Embedding,
    RMSNorm,
    BaselineMLP,
    RMSNorm,
    BaselineAttention,
]:
    return DecoderFactory(
        embedding_factory=EmbeddingFactory(precision=precision),
        rope_factory=RoPEFactory(precision=precision),
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
        out_norm_factory=RMSNormFactory(
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
    rope_params: RoPEParams,
    eps: float,
    key: PRNGKeyArray,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> BaselineLlama:
    factory = get_baseline_llama_factory(
        precision=precision,
        accumulation_precision=accumulation_precision,
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
        rope_params=rope_params,
        eps=eps,
        key=key,
    )
