
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike, Array
from collections.abc import Mapping

from torch._dynamo.utils import istype

from lalamo.modules import (
    AttentionConfig,
    TransformerLayerConfig,
    FullPrecisionLinearConfig,
    UnscaledRoPEConfig,
    UpcastMode,

    ClassifierConfig,
    NormalizationConfig,
    DenseMLPConfig,
    TransformerConfig
)
from lalamo.modules.activations import GELU, SiLU
from lalamo.modules.classifier import PredictionHeadConfig, activation_from_str
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

    def to_classifier_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike) -> ClassifierConfig:

        # TODO: could not find default value for this one in Mirai's transformer
        embedding_config = UntiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
                precision=activation_precision,
        )

        # TODO: looks like using 'default' type of Rope scaling which means unscaled - UnscaledRoPEConfig
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.global_rope_theta,
            max_sequence_length=context_length or self.max_position_embeddings,
        )

        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.layer_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=self.attention_bias,
            has_out_biases=False,
        )
        activation = activation_from_str(self.hidden_activation)
        assert activation is SiLU or activation is GELU
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=activation,
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )
        decoder_layer_config = TransformerLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None
        )

        transformer_config = TransformerConfig(
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=1,
            head_dim=self.hidden_size // self.num_attention_heads,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=None,
            context_length=context_length or self.max_position_embeddings,
        )

        prediction_head_config = PredictionHeadConfig(
            input_size = self.hidden_size,
            output_size = self.hidden_size,
            use_bias = self.classifier_bias,
            activation = activation_from_str(self.classifier_activation),
            norm_size = self.hidden_size,
            norm_eps = self.norm_eps,
            use_norm_bias = self.norm_bias
        )

        final_linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )

        return ClassifierConfig(
            embedding_config = embedding_config,
            transformer_config = transformer_config,
            prediction_head_config = prediction_head_config,
            final_linear_config = final_linear_config,
            vocab_size = self.vocab_size,
            model_dim = self.hidden_size,
            hidden_dim = self.hidden_size,
            num_heads = self.num_attention_heads,
            # num_groups: int  NOTE: this one seem to be not used in ModertBert attention
            head_dim = self.hidden_size // self.num_attention_heads,
            attention_scale = None,
            num_layers = self.num_hidden_layers,
            sliding_window_sizes = None, # TODO: figure out this one from self.local_attention
            context_length = self.max_position_embeddings,
            num_labels = len(self.label2id)
        )