from fartsovka.common import DType
from fartsovka.modules.activations import Activation
from fartsovka.modules.attention import Attention, AttentionConfig
from fartsovka.modules.decoder import Decoder, DecoderConfig
from fartsovka.modules.decoder_layer import PreNormDecoderLayer, PreNormDecoderLayerConfig
from fartsovka.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from fartsovka.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from fartsovka.modules.mlp import MLP, MLPConfig
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.rope import LlamaRoPE, LlamaRoPEConfig

__all__ = [
    "LlamaAttention",
    "LlamaConfig",
    "LlamaDecoder",
    "LlamaDecoderLayer",
    "LlamaMLP",
]

type LlamaMLP = MLP[FullPrecisionLinear]

type LlamaAttention = Attention[FullPrecisionLinear, FullPrecisionLinear]

type LlamaDecoderLayer = PreNormDecoderLayer[
    RMSNorm,
    LlamaMLP,
    RMSNorm,
    LlamaAttention,
]

type LlamaDecoder = Decoder[
    TiedEmbedding,
    LlamaDecoderLayer,
    LlamaRoPE,
]


type LlamaDecoderConfig = DecoderConfig[
    TiedEmbedding,
    LlamaDecoderLayer,
    LlamaRoPE,
]


class LlamaConfig(DecoderConfig[TiedEmbedding, LlamaDecoderLayer, LlamaRoPE]):
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
        precision: DType,
        accumulation_precision: DType,
        rope_scaling_factor: float,
        original_context_length: int,
        rope_low_frequency_factor: float,
        rope_high_frequency_factor: float,
    ) -> None:
        super().__init__(
            embedding_config=TiedEmbeddingConfig(precision=precision),
            rope_config=LlamaRoPEConfig(
                precision=precision,
                scaling_factor=rope_scaling_factor,
                original_context_length=original_context_length,
                low_frequency_factor=rope_low_frequency_factor,
                high_frequency_factor=rope_high_frequency_factor,
            ),
            layer_config=PreNormDecoderLayerConfig(
                attention_norm_config=RMSNormConfig(
                    scale_precision=precision,
                    accumulation_precision=accumulation_precision,
                ),
                attention_config=AttentionConfig(
                    qkv_projection_config=FullPrecisionLinearConfig(precision=precision),
                    out_projection_config=FullPrecisionLinearConfig(precision=precision),
                ),
                mlp_norm_config=RMSNormConfig(
                    scale_precision=precision,
                    accumulation_precision=accumulation_precision,
                ),
                mlp_config=MLPConfig(
                    linear_config=FullPrecisionLinearConfig(precision=precision),
                ),
            ),
            output_norm_config=RMSNormConfig(
                scale_precision=precision,
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
            rope_theta=rope_theta,
            max_sequence_length=max_sequence_length,
            eps=eps,
        )
