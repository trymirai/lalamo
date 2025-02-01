from fartsovka.common import DType
from fartsovka.modules.activations import Activation
from fartsovka.modules.attention import Attention, AttentionConfig
from fartsovka.modules.decoder import Decoder, DecoderConfig
from fartsovka.modules.decoder_layer import PreNormDecoderLayer, PreNormDecoderLayerConfig
from fartsovka.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from fartsovka.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from fartsovka.modules.mlp import MLP, MLPConfig
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.rope import AbstractRoPE, RoPEConfig

__all__ = [
    "Qwen2Config",
    "Qwen2Decoder",
    "Qwen2Attention",
    "Qwen2DecoderLayer",
    "Qwen2MLP",
]

type Qwen2MLP = MLP[FullPrecisionLinear]

type Qwen2Attention = Attention[FullPrecisionLinear, FullPrecisionLinear]

type Qwen2DecoderLayer = PreNormDecoderLayer[
    RMSNorm,
    Qwen2MLP,
    RMSNorm,
    Qwen2Attention,
]

type Qwen2Decoder = Decoder[
    TiedEmbedding,
    Qwen2DecoderLayer,
    AbstractRoPE,
]


QWEN_BETA_FAST = 32.0
QWEN_BETA_SLOW = 1.0
QWEN_ROPE_SCALING_FACTOR = 4.0


class Qwen2Config(DecoderConfig[TiedEmbedding, Qwen2DecoderLayer, AbstractRoPE]):
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
        sliding_window_sizes: list[int | None] | None,
        rope_theta: float,
        eps: float,
        precision: DType,
        accumulation_precision: DType,
    ) -> None:
        super().__init__(
            embedding_config=TiedEmbeddingConfig(precision=precision),
            rope_config=RoPEConfig(
                precision=precision,
                # scaling_factor=QWEN_ROPE_SCALING_FACTOR,  # noqa: ERA001
                # beta_fast=QWEN_BETA_FAST,  # noqa: ERA001
                # beta_slow=QWEN_BETA_SLOW,  # noqa: ERA001
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
            use_attention_qkv_bias=True,
            use_attention_out_bias=False,
            attention_scale=None,
            attention_logit_soft_cap=None,
            sliding_window_sizes=sliding_window_sizes,
            rope_theta=rope_theta,
            max_sequence_length=max_sequence_length,
            eps=eps,
        )
