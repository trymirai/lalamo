
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike, Array
from collections.abc import Mapping

from lalamo.modules import (
    Activation,
    AttentionConfig,
    DecoderLayerConfig,
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    LlamaRoPEConfig,
    MLPConfig,
    TiedEmbeddingConfig,
    UnscaledRoPEConfig,
    UpcastMode,
    ModernBertPredictionHeadConfig,
    LinearConfig,
    Decoder
)
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.quantization import QuantizationMode

from .common import AWQQuantizationConfig, GPTQQuantizationConfig, HuggingFaceConfig

__all__ = ["ModernBERTConfig"]


@dataclass(frozen=True)
class LlamaRopeScalingConfig:
    factor: float
    high_freq_factor: float
    low_freq_factor: float
    original_max_position_embeddings: int
    rope_type: Literal["llama3"]


@dataclass(frozen=True)
class ModernBERTConfig(HuggingFaceConfig):
    architectures: list[Literal["ModernBertForSequenceClassification"]]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int | list[int]
    classifier_activation: Literal["gelu"]
    classifier_bias: bool
    classifier_dropout: float
    classifier_pooling: Literal["mean"]
    cls_token_id: int | list[int]
    decoder_bias: bool
    deterministic_flash_attn: bool
    embedding_dropout: float
    eos_token_id: int | list[int]
    global_attn_every_n_layers: int
    global_rope_theta: float
    gradient_checkpointing: bool
    hidden_activation: Literal["gelu"]
    hidden_size: int
    initializer_cutoff_factor: float
    initializer_range: float
    intermediate_size: int
    layer_norm_eps:float
    local_attention: int
    local_rope_theta: float
    max_position_embeddings: int
    mlp_bias: bool
    mlp_dropout: float
    model_type: Literal["modernbert"]
    norm_bias: bool
    norm_eps: float
    num_attention_heads: int
    num_hidden_layers: int
    pad_token_id: int | list[int]
    position_embedding_type: Literal["absolute"]
    sep_token_id: int | list[int]
    transformers_version: str
    vocab_size: int
    id2label: dict[str, str]
    label2id: dict[str, int]

    # NOTE: this one is present in vanilla modern-bert from HF (answerdotai/ModernBERT-base)
    # tie_word_embeddings: bool
    
    quantization_config: AWQQuantizationConfig | GPTQQuantizationConfig | None = None
