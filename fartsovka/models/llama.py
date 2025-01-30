from dataclasses import dataclass

from jaxtyping import PRNGKeyArray

from fartsovka.common import DType
from fartsovka.modules.activations import Activation
from fartsovka.modules.attention import Attention, AttentionConfig
from fartsovka.modules.decoder import Decoder, DecoderConfig, DecoderConfigType
from fartsovka.modules.decoder_layer import DecoderLayer, DecoderLayerConfig
from fartsovka.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from fartsovka.modules.linear import Linear, LinearConfig
from fartsovka.modules.mlp import MLP, MLPConfig
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.rope import LlamaRoPE, LlamaRoPEConfig

from .common import AbstractModelConfig

__all__ = [
    "LlamaAttention",
    "LlamaConfig",
    "LlamaDecoder",
    "LlamaDecoderLayer",
    "LlamaMLP",
]

type LlamaMLP = MLP[Linear]

type LlamaAttention = Attention[Linear, Linear]

type LlamaDecoderLayer = DecoderLayer[
    RMSNorm,
    LlamaMLP,
    RMSNorm,
    LlamaAttention,
]

type LlamaDecoder = Decoder[
    TiedEmbedding,
    RMSNorm,
    LlamaMLP,
    RMSNorm,
    LlamaAttention,
    LlamaRoPE,
]


type LlamaDecoderConfig = DecoderConfig[
    TiedEmbedding,
    RMSNorm,
    LlamaMLP,
    RMSNorm,
    LlamaAttention,
    LlamaRoPE,
]


@dataclass
class LlamaConfig(AbstractModelConfig[LlamaDecoder]):
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
    decoder_config: DecoderConfigType

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
        self.decoder_config = _get_llama_decoder_config(
            precision=precision,
            accumulation_precision=accumulation_precision,
            rope_scaling_factor=rope_scaling_factor,
            original_context_length=original_context_length,
            rope_low_frequency_factor=rope_low_frequency_factor,
            rope_high_frequency_factor=rope_high_frequency_factor,
        )

    def __call__(self, key: PRNGKeyArray) -> LlamaDecoder:
        return self.decoder_config(  # type: ignore
            num_layers=self.num_layers,
            vocab_dim=self.vocab_dim,
            model_dim=self.model_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            activation=Activation.SILU,
            use_mlp_bias=False,
            use_attention_qkv_bias=False,
            use_attention_out_bias=False,
            sliding_window_sizes=None,
            max_sequence_length=self.max_sequence_length,
            rope_theta=self.rope_theta,
            eps=self.eps,
            key=key,
        )


def _get_llama_decoder_config(
    *,
    precision: DType,
    accumulation_precision: DType,
    rope_scaling_factor: float,
    original_context_length: int,
    rope_low_frequency_factor: float,
    rope_high_frequency_factor: float,
) -> LlamaDecoderConfig:
    return DecoderConfig(
        embedding_config=TiedEmbeddingConfig(precision=precision),
        rope_config=LlamaRoPEConfig(
            precision=precision,
            scaling_factor=rope_scaling_factor,
            original_context_length=original_context_length,
            low_frequency_factor=rope_low_frequency_factor,
            high_frequency_factor=rope_high_frequency_factor,
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
