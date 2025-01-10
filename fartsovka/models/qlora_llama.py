from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.modules.attention import Attention, AttentionFactory
from fartsovka.modules.decoder import Decoder, DecoderFactory
from fartsovka.modules.decoder_layer import DecoderLayer, DecoderLayerFactory
from fartsovka.modules.embedding import QuantizedEmbedding, QuantizedEmbeddingFactory
from fartsovka.modules.linear import QLoRALinear, QLoRALinearFactory
from fartsovka.modules.mlp import MLP, MLPFactory
from fartsovka.modules.normalization import RMSNorm, RMSNormFactory
from fartsovka.modules.rope import RoPEFactory, RoPEParams
from fartsovka.quantization import QuantizationMode

__all__ = [
    "QLoRAMLP",
    "QLoRAAttention",
    "QLoRADecoderLayer",
    "QLoRALlama",
    "get_qlora_llama",
    "get_qlora_llama_factory",
]

type QLoRAMLP = MLP[QLoRALinear]

type QLoRAAttention = Attention[QLoRALinear, QLoRALinear]

type QLoRADecoderLayer = DecoderLayer[
    RMSNorm,
    QLoRAMLP,
    RMSNorm,
    QLoRAAttention,
]

type QLoRALlama = Decoder[
    QuantizedEmbedding,
    RMSNorm,
    QLoRAMLP,
    RMSNorm,
    QLoRAAttention,
]


def get_qlora_llama_factory(
    *,
    weight_quantization_mode: QuantizationMode,
    embedding_quantization_mode: QuantizationMode,
    quantization_group_size: int,
    lora_rank: int,
    lora_scale: float = 2.0,
    activation_precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> DecoderFactory[
    QuantizedEmbedding,
    RMSNorm,
    QLoRAMLP,
    RMSNorm,
    QLoRAAttention,
]:
    return DecoderFactory(
        embedding_factory=QuantizedEmbeddingFactory(
            mode=embedding_quantization_mode,
            activation_precision=activation_precision,
        ),
        rope_factory=RoPEFactory(precision=activation_precision),
        layer_factory=DecoderLayerFactory(
            attention_norm_factory=RMSNormFactory(
                precision=activation_precision,
                accumulation_precision=accumulation_precision,
            ),
            attention_factory=AttentionFactory(
                qkv_projection_factory=QLoRALinearFactory(
                    group_size=quantization_group_size,
                    mode=weight_quantization_mode,
                    lora_rank=lora_rank,
                    lora_scale=lora_scale,
                ),
                out_projection_factory=QLoRALinearFactory(
                    group_size=quantization_group_size,
                    mode=weight_quantization_mode,
                    lora_rank=lora_rank,
                    lora_scale=lora_scale,
                ),
            ),
            mlp_norm_factory=RMSNormFactory(
                precision=activation_precision,
                accumulation_precision=accumulation_precision,
            ),
            mlp_factory=MLPFactory(
                linear_factory=QLoRALinearFactory(
                    group_size=quantization_group_size,
                    mode=weight_quantization_mode,
                    lora_rank=lora_rank,
                    lora_scale=lora_scale,
                ),
            ),
        ),
        out_norm_factory=RMSNormFactory(
            precision=activation_precision,
            accumulation_precision=accumulation_precision,
        ),
    )


def get_qlora_llama(
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
    weight_quantization_mode: QuantizationMode,
    embedding_quantization_mode: QuantizationMode,
    quantization_group_size: int,
    lora_rank: int,
    lora_scale: float = 2.0,
    activation_precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> QLoRALlama:
    factory = get_qlora_llama_factory(
        weight_quantization_mode=weight_quantization_mode,
        embedding_quantization_mode=embedding_quantization_mode,
        quantization_group_size=quantization_group_size,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        activation_precision=activation_precision,
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
