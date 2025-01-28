from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.modules.attention import Attention, AttentionConfig
from fartsovka.modules.decoder import Decoder, DecoderConfig
from fartsovka.modules.decoder_layer import DecoderLayer, DecoderLayerConfig
from fartsovka.modules.embedding import QuantizedEmbedding, QuantizedEmbeddingConfig
from fartsovka.modules.linear import QLoRALinear, QLoRALinearConfig
from fartsovka.modules.mlp import MLP, MLPConfig
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.rope import LlamaRoPE, LlamaRoPEConfig
from fartsovka.quantization import QuantizationMode

__all__ = [
    "get_qlora_llama_factory",
    "get_qlora_llama",
    "QLoRALlamaAttention",
    "QLoRALlamaDecoder",
    "QLoRALlamaDecoderLayer",
    "QLoRALlamaMLP",
]

type QLoRALlamaMLP = MLP[QLoRALinear]

type QLoRALlamaAttention = Attention[QLoRALinear, QLoRALinear]

type QLoRALlamaDecoderLayer = DecoderLayer[
    RMSNorm,
    QLoRALlamaMLP,
    RMSNorm,
    QLoRALlamaAttention,
]

type QLoRALlamaDecoder = Decoder[
    QuantizedEmbedding,
    RMSNorm,
    QLoRALlamaMLP,
    RMSNorm,
    QLoRALlamaAttention,
    LlamaRoPE,
]


def get_qlora_llama_factory(
    *,
    weight_quantization_mode: QuantizationMode,
    embedding_quantization_mode: QuantizationMode,
    activation_quantization_mode: QuantizationMode | None,
    quantization_group_size: int,
    lora_rank: int,
    lora_scale: float = 2.0,
    activation_precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
    rope_scaling_factor: float,
    rope_original_context_length: int,
    rope_low_frequency_factor: float,
    rope_high_frequency_factor: float,
) -> DecoderConfig[
    QuantizedEmbedding,
    RMSNorm,
    QLoRALlamaMLP,
    RMSNorm,
    QLoRALlamaAttention,
    LlamaRoPE,
]:
    return DecoderConfig(
        embedding_config=QuantizedEmbeddingConfig(
            embedding_quantization_mode=embedding_quantization_mode,
            activation_precision=activation_precision,
            activation_quantization_mode=activation_quantization_mode,
        ),
        rope_config=LlamaRoPEConfig(
            precision=activation_precision,
            scaling_factor=rope_scaling_factor,
            original_context_length=rope_original_context_length,
            low_frequency_factor=rope_low_frequency_factor,
            high_frequency_factor=rope_high_frequency_factor,
        ),
        layer_config=DecoderLayerConfig(
            attention_norm_config=RMSNormConfig(
                precision=activation_precision,
                accumulation_precision=accumulation_precision,
            ),
            attention_config=AttentionConfig(
                qkv_projection_config=QLoRALinearConfig(
                    group_size=quantization_group_size,
                    weight_quantization_mode=weight_quantization_mode,
                    activation_quantization_mode=activation_quantization_mode,
                    activation_precision=activation_precision,
                    lora_rank=lora_rank,
                    lora_scale=lora_scale,
                ),
                out_projection_config=QLoRALinearConfig(
                    group_size=quantization_group_size,
                    weight_quantization_mode=weight_quantization_mode,
                    activation_quantization_mode=activation_quantization_mode,
                    activation_precision=activation_precision,
                    lora_rank=lora_rank,
                    lora_scale=lora_scale,
                ),
            ),
            mlp_norm_config=RMSNormConfig(
                precision=activation_precision,
                accumulation_precision=accumulation_precision,
            ),
            mlp_config=MLPConfig(
                linear_config=QLoRALinearConfig(
                    group_size=quantization_group_size,
                    weight_quantization_mode=weight_quantization_mode,
                    activation_quantization_mode=activation_quantization_mode,
                    activation_precision=activation_precision,
                    lora_rank=lora_rank,
                    lora_scale=lora_scale,
                ),
            ),
        ),
        output_norm_config=RMSNormConfig(
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
    rope_theta: float,
    rope_scaling_factor: float,
    rope_original_context_length: int,
    rope_low_frequency_factor: float,
    rope_high_frequency_factor: float,
    eps: float,
    key: PRNGKeyArray,
    weight_quantization_mode: QuantizationMode,
    embedding_quantization_mode: QuantizationMode,
    activation_quantization_mode: QuantizationMode | None,
    quantization_group_size: int,
    lora_rank: int,
    lora_scale: float = 2.0,
    activation_precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> QLoRALlamaDecoder:
    factory = get_qlora_llama_factory(
        weight_quantization_mode=weight_quantization_mode,
        embedding_quantization_mode=embedding_quantization_mode,
        activation_quantization_mode=activation_quantization_mode,
        quantization_group_size=quantization_group_size,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        activation_precision=activation_precision,
        accumulation_precision=accumulation_precision,
        rope_scaling_factor=rope_scaling_factor,
        rope_original_context_length=rope_original_context_length,
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
        use_attention_qkv_bias=False,
        use_attention_out_bias=False,
        use_mlp_bias=False,
        max_sequence_length=max_sequence_length,
        rope_theta=rope_theta,
        eps=eps,
        key=key,
    )
