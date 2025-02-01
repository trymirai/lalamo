from fartsovka.common import DType
from fartsovka.modules.activations import Activation
from fartsovka.modules.attention import AttentionConfig
from fartsovka.modules.decoder_layer import PreNormDecoderLayerConfig
from fartsovka.modules.embedding import TiedEmbeddingConfig
from fartsovka.modules.linear import FullPrecisionLinearConfig
from fartsovka.modules.mlp import MLPConfig
from fartsovka.modules.normalization import RMSNormConfig
from fartsovka.modules.rope import LlamaRoPEConfig

__all__ = [
    "get_llama_config",
]


def get_llama_config(
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
