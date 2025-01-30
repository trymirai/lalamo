from fartsovka.common import DType
from fartsovka.modules.activations import Activation
from fartsovka.modules.attention import Attention, AttentionConfig
from fartsovka.modules.decoder import Decoder, DecoderConfig
from fartsovka.modules.decoder_layer import PreNormDecoderLayer, PreNormDecoderLayerConfig
from fartsovka.modules.embedding import QuantizedTiedEmbedding, QuantizedTiedEmbeddingConfig
from fartsovka.modules.linear import QLoRALinear, QLoRALinearConfig
from fartsovka.modules.mlp import MLP, MLPConfig
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.rope import LlamaRoPE, LlamaRoPEConfig
from fartsovka.quantization import QuantizationMode

__all__ = [
    "QLoRALlamaConfig",
    "QLoRALlamaAttention",
    "QLoRALlamaDecoder",
    "QLoRALlamaDecoderLayer",
    "QLoRALlamaMLP",
]

type QLoRALlamaMLP = MLP[QLoRALinear]

type QLoRALlamaAttention = Attention[QLoRALinear, QLoRALinear]

type QLoRALlamaDecoderLayer = PreNormDecoderLayer[
    RMSNorm,
    QLoRALlamaMLP,
    RMSNorm,
    QLoRALlamaAttention,
]

type QLoRALlamaDecoder = Decoder[
    QuantizedTiedEmbedding,
    QLoRALlamaDecoderLayer,
    LlamaRoPE,
]


class QLoRALlamaConfig(
    DecoderConfig[
        QuantizedTiedEmbedding,
        QLoRALlamaDecoderLayer,
        LlamaRoPE,
    ],
):
    def __init__(
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
        rope_theta: float,
        eps: float,
        weight_quantization_mode: QuantizationMode,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: QuantizationMode | None,
        quantization_group_size: int,
        lora_rank: int,
        lora_scale: float,
        activation_precision: DType,
        accumulation_precision: DType,
        rope_scaling_factor: float,
        original_context_length: int,
        rope_low_frequency_factor: float,
        rope_high_frequency_factor: float,
    ) -> None:
        super().__init__(
            embedding_config=QuantizedTiedEmbeddingConfig(
                embedding_quantization_mode=embedding_quantization_mode,
                activation_precision=activation_precision,
                activation_quantization_mode=activation_quantization_mode,
            ),
            rope_config=LlamaRoPEConfig(
                precision=activation_precision,
                scaling_factor=rope_scaling_factor,
                original_context_length=original_context_length,
                low_frequency_factor=rope_low_frequency_factor,
                high_frequency_factor=rope_high_frequency_factor,
            ),
            layer_config=PreNormDecoderLayerConfig(
                attention_norm_config=RMSNormConfig(
                    scale_precision=activation_precision,
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
                    scale_precision=activation_precision,
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
                scale_precision=activation_precision,
                accumulation_precision=accumulation_precision,
            ),
            num_layers=num_layers,
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            embedding_input_scale=None,
            output_logits_soft_cap=None,
            activation=Activation.SILU,
            use_mlp_bias=False,
            use_attention_qkv_bias=False,
            use_attention_out_bias=False,
            attention_scale=None,
            attention_logit_soft_cap=None,
            sliding_window_sizes=None,
            max_sequence_length=max_sequence_length,
            rope_theta=rope_theta,
            eps=eps,
        )
