from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from .attention import Attention, AttentionFactory
from .common import DEFAULT_PRECISION
from .decoder import Decoder, DecoderFactory
from .decoder_layer import DecoderLayer, DecoderLayerFactory
from .embedding import EmbeddingFactory
from .linear import Linear, LinearFactory
from .mlp import MLP, MLPFactory
from .normalisation import RMSNorm, RMSNormFactory
from .rope import RoPEFactory

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
    RMSNorm,
    BaselineMLP,
    RMSNorm,
    BaselineAttention,
]


def get_baseline_llama_factory(
    precision: jnp.dtype = DEFAULT_PRECISION,
    accumulation_precision: jnp.dtype = jnp.float32,
) -> DecoderFactory[
    RMSNorm,
    BaselineMLP,
    RMSNorm,
    BaselineAttention,
]:
    return DecoderFactory(
        embedding_factory=EmbeddingFactory(precision=precision),
        rope_factory=RoPEFactory(precision=precision),
        layer_factory=DecoderLayerFactory(
            pre_attention_norm_factory=RMSNormFactory(
                precision=precision,
                accumulation_precision=accumulation_precision,
            ),
            attention_factory=AttentionFactory(
                qkv_projection_factory=LinearFactory(precision=precision),
                out_projection_factory=LinearFactory(precision=precision),
            ),
            pre_mlp_norm_factory=RMSNormFactory(
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
    precision: jnp.dtype = DEFAULT_PRECISION,
    accumulation_precision: jnp.dtype = jnp.float32,
) -> BaselineLlama:
    factory = get_baseline_llama_factory(
        precision=precision,
        accumulation_precision=accumulation_precision,
    )
    return factory(
        num_layers,
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
