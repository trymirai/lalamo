from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jaxtyping import DTypeLike

from lalamo.modules import (
    Activation,
    AttentionConfig,
    ClassifierConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    NormalizationConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.activations import GELU, SiLU
from lalamo.modules.classifier import (
    PoolingType,
    PredictionHeadConfig,
)
from lalamo.modules.embedding import TiedEmbeddingConfig

from .common import (
    AWQQuantizationConfig,
    GPTQQuantizationConfig,
    HuggingFaceClassifierConfig,
)

__all__ = ["ModernBERTConfig"]


def activation_from_str(activation: str) -> type[Activation]:
    supported_activations = {
        "silu": SiLU,
        "gelu": GELU,
    }
    if activation in supported_activations:
        return supported_activations[activation]

    raise ValueError(
        f"Only activations from the following list are supported by Classifier: {supported_activations.keys()}",
    )


@dataclass(frozen=True)
class ModernBERTConfig(HuggingFaceClassifierConfig):
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
    layer_norm_eps: float
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
    id2label: dict[int, str]
    label2id: dict[str, int]

    quantization_config: AWQQuantizationConfig | GPTQQuantizationConfig | None = None

    def __post_init__(self) -> None:
        if len(self.label2id) != len(self.id2label):
            raise ValueError("Legnth of label2id and id2label is expected to be the same")

    def calculate_sliding_windows(self, num_layers: int, global_attn_every_n_layers: int) -> tuple[None, ...]:
        result = [None] * num_layers
        for index in range(len(result)):
            if index % global_attn_every_n_layers != 0:
                result[index] = self.local_attention
            else:
                pass
        return tuple(result)

    def to_classifier_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> ClassifierConfig:
        embedding_config = TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=activation_precision,
        )
        embedding_norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True,
        )

        global_rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.global_rope_theta,
            max_sequence_length=context_length or self.max_position_embeddings,
        )
        local_rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.local_rope_theta,
            max_sequence_length=context_length or self.max_position_embeddings,
        )

        sliding_window_sizes = self.calculate_sliding_windows(self.num_hidden_layers, self.global_attn_every_n_layers)

        transformer_norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        activation = activation_from_str(self.hidden_activation)
        assert activation is SiLU or activation is GELU
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=activation(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        # In ModernBERT architecture first Transformer layer has no pre-attention normalization
        pre_attn_configs = [transformer_norm_config if i > 0 else None for i in range(self.num_hidden_layers)]

        transformer_layer_configs = []
        for sliding_window_size, pre_attn_config in zip(sliding_window_sizes, pre_attn_configs, strict=True):
            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=self.attention_bias,
                has_out_biases=False,
                num_heads=self.num_attention_heads,
                num_groups=self.num_attention_heads,
                head_dim=self.hidden_size // self.num_attention_heads,
                scale=None,
                is_causal=False,
                sliding_window_size=sliding_window_size,
            )
            layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=pre_attn_config,
                mixer_config=attention_config,
                post_mixer_norm_config=None,
                pre_mlp_norm_config=transformer_norm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
            )
            transformer_layer_configs.append(layer_config)

        transformer_config = TransformerConfig(
            global_rope_config=global_rope_config,
            local_rope_config=local_rope_config,
            layer_configs=tuple(transformer_layer_configs),
            output_norm_config=transformer_norm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=context_length or self.max_position_embeddings,
        )

        prediction_head_dense_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        prediction_head_norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=jnp.float32,
            epsilon=self.norm_eps,
            scale_offset=0.0,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True,
        )
        prediction_head_activation = activation_from_str(self.classifier_activation)
        prediction_head_readout_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        prediction_head_config = PredictionHeadConfig(
            dense_config=prediction_head_dense_config,
            activation=prediction_head_activation(),
            normalization_config=prediction_head_norm_config,
            readout_config=prediction_head_readout_config,
            use_dense_bias=self.classifier_bias,
        )

        output_labels = [self.id2label[idx] for idx in range(len(self.id2label))]

        return ClassifierConfig(
            embedding_config=embedding_config,
            embedding_norm_config=embedding_norm_config,
            transformer_config=transformer_config,
            prediction_head_config=prediction_head_config,
            readout_config=prediction_head_readout_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            context_length=self.max_position_embeddings,
            num_labels=len(self.id2label),
            classifier_pooling=PoolingType(self.classifier_pooling),
            output_labels=tuple(output_labels),
        )
