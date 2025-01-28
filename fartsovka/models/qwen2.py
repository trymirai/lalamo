from dataclasses import dataclass

from jaxtyping import PRNGKeyArray

from fartsovka.common import DType
from fartsovka.modules.attention import Attention, AttentionConfig
from fartsovka.modules.decoder import Decoder, DecoderConfig
from fartsovka.modules.decoder_layer import DecoderLayer, DecoderLayerConfig
from fartsovka.modules.embedding import Embedding, EmbeddingConfig
from fartsovka.modules.linear import Linear, LinearConfig
from fartsovka.modules.mlp import MLP, MLPConfig
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.rope import AbstractRoPE, RoPEConfig

from .common import AbstractModelConfig

__all__ = [
    "Qwen2Config",
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

type Qwen2DecoderConfig = DecoderConfig[
    Embedding,
    RMSNorm,
    Qwen2MLP,
    RMSNorm,
    Qwen2Attention,
    AbstractRoPE,
]


@dataclass
class Qwen2Config(AbstractModelConfig[Qwen2Decoder]):
    num_layers: int
    vocab_dim: int
    model_dim: int
    hidden_dim: int
    num_heads: int
    num_groups: int
    head_dim: int
    max_sequence_length: int
    rope_theta: float
    eps: float
    decoder_config: Qwen2DecoderConfig

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
        rope_beta_fast: float,
        rope_beta_slow: float,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.rope_theta = rope_theta
        self.eps = eps
        self.decoder_config = _get_qwen2_decoder_config(
            precision=precision,
            accumulation_precision=accumulation_precision,
            rope_scaling_factor=rope_scaling_factor,
            rope_beta_fast=rope_beta_fast,
            rope_beta_slow=rope_beta_slow,
        )

    def __call__(self, key: PRNGKeyArray) -> Qwen2Decoder:
        return self.decoder_config(
            num_layers=self.num_layers,
            vocab_dim=self.vocab_dim,
            model_dim=self.model_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            use_attention_qkv_bias=True,
            use_attention_out_bias=False,
            use_mlp_bias=False,
            max_sequence_length=self.max_sequence_length,
            rope_theta=self.rope_theta,
            eps=self.eps,
            key=key,
        )


def _get_qwen2_decoder_config(
    *,
    precision: DType,
    accumulation_precision: DType,
    rope_scaling_factor: float,  # noqa: ARG001
    rope_beta_fast: float,  # noqa: ARG001
    rope_beta_slow: float,  # noqa: ARG001
) -> Qwen2DecoderConfig:
    return DecoderConfig(
        embedding_config=EmbeddingConfig(precision=precision),
        rope_config=RoPEConfig(
            precision=precision,
            # scaling_factor=rope_scaling_factor,  # noqa: ERA001
            # beta_fast=rope_beta_fast,  # noqa: ERA001
            # beta_slow=rope_beta_slow,  # noqa: ERA001
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
