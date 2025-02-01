from fartsovka.common import DType
from fartsovka.modules.activations import Activation
from fartsovka.modules.attention import Attention, AttentionConfig
from fartsovka.modules.decoder import Decoder, DecoderConfig
from fartsovka.modules.decoder_layer import PrePostNormDecoderLayer, PrePostNormDecoderLayerConfig
from fartsovka.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from fartsovka.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from fartsovka.modules.mlp import MLP, MLPConfig
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.rope import AbstractRoPE, RoPEConfig

__all__ = [
    "Gemma2Attention",
    "Gemma2Config",
    "Gemma2Decoder",
    "Gemma2DecoderLayer",
    "Gemma2MLP",
]

type Gemma2MLP = MLP[FullPrecisionLinear]

type Gemma2Attention = Attention[FullPrecisionLinear, FullPrecisionLinear]

type Gemma2DecoderLayer = PrePostNormDecoderLayer[
    RMSNorm,
    Gemma2MLP,
    RMSNorm,
    RMSNorm,
    Gemma2Attention,
    RMSNorm,
]

type Gemma2Decoder = Decoder[
    TiedEmbedding,
    Gemma2DecoderLayer,
    AbstractRoPE,
]


type Gemma2DecoderConfig = DecoderConfig[
    TiedEmbedding,
    Gemma2DecoderLayer,
    AbstractRoPE,
]


class Gemma2Config(DecoderConfig[TiedEmbedding, Gemma2DecoderLayer, AbstractRoPE]):
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
        output_logits_soft_cap: float,
        max_sequence_length: int,
        sliding_window_size: int,
        query_pre_attn_scalar: float,
        attention_logits_soft_cap: float,
        rope_theta: float,
        eps: float,
        precision: DType,
        accumulation_precision: DType,
    ) -> None:
        sliding_window_sizes = [sliding_window_size if not bool(i % 2) else None for i in range(num_layers)]
        embedding_input_scale = model_dim**0.5
        attention_scale = query_pre_attn_scalar**-0.5
        super().__init__(
            embedding_config=TiedEmbeddingConfig(precision=precision),
            rope_config=RoPEConfig(
                precision=precision,
            ),
            layer_config=PrePostNormDecoderLayerConfig(
                attention_pre_norm_config=RMSNormConfig(
                    scale_precision=accumulation_precision,
                    accumulation_precision=accumulation_precision,
                ),
                attention_config=AttentionConfig(
                    qkv_projection_config=FullPrecisionLinearConfig(precision=precision),
                    out_projection_config=FullPrecisionLinearConfig(precision=precision),
                ),
                attention_post_norm_config=RMSNormConfig(
                    scale_precision=accumulation_precision,
                    accumulation_precision=accumulation_precision,
                ),
                mlp_pre_norm_config=RMSNormConfig(
                    scale_precision=accumulation_precision,
                    accumulation_precision=accumulation_precision,
                ),
                mlp_config=MLPConfig(
                    linear_config=FullPrecisionLinearConfig(precision=precision),
                ),
                mlp_post_norm_config=RMSNormConfig(
                    scale_precision=accumulation_precision,
                    accumulation_precision=accumulation_precision,
                ),
            ),
            output_norm_config=RMSNormConfig(
                scale_precision=accumulation_precision,
                accumulation_precision=accumulation_precision,
            ),
            num_layers=num_layers,
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            embedding_input_scale=embedding_input_scale,
            output_logits_soft_cap=output_logits_soft_cap,
            activation=Activation.GELU,
            use_mlp_bias=False,
            use_attention_qkv_bias=False,
            use_attention_out_bias=False,
            attention_scale=attention_scale,
            attention_logit_soft_cap=attention_logits_soft_cap,
            sliding_window_sizes=sliding_window_sizes,
            rope_theta=rope_theta,
            max_sequence_length=max_sequence_length,
            eps=eps,
        )
