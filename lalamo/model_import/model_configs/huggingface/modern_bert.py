from dataclasses import dataclass
from typing import Literal

from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.loaders.huggingface import load_modern_bert_attention, load_modern_bert_mlp
from lalamo.modules.activations import GELU
from lalamo.modules.classifier import (
    ClassifierConfig,
    PoolingType,
    PredictionHeadConfig,
)
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.flow import EmitConfig, SequenceConfig, StepConfig, residual
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.utils.parameter_path import ParameterPath

from .common import (
    AWQQuantizationConfig,
    GPTQQuantizationConfig,
    HuggingFaceClassifierConfig,
)

__all__ = ["ModernBERTConfig"]


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
            raise ValueError("Length of label2id and id2label is expected to be the same")

    def calculate_sliding_windows(self) -> tuple[int | None, ...]:
        return tuple(
            None if layer_index % self.global_attn_every_n_layers == 0 else self.local_attention
            for layer_index in range(self.num_hidden_layers)
        )

    def build_classifier(self, at: Binder) -> ClassifierConfig:
        head_dim = self.hidden_size // self.num_attention_heads
        embedding_config = TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
        )
        embedding_norm_config = NormalizationConfig(
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True,
        )

        global_rope_config = UnscaledRoPEConfig(
            base=self.global_rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=head_dim,
        )
        local_rope_config = UnscaledRoPEConfig(
            base=self.local_rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=head_dim,
        )

        sliding_window_sizes = self.calculate_sliding_windows()

        transformer_norm_config = NormalizationConfig(
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True,
        )
        linear_config = LinearConfig()
        activation = GELU(approximate=False)
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=activation,
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
            hidden_dim=self.intermediate_size,
        )

        steps: list[StepConfig] = [at("model.embeddings.norm", embedding_norm_config)]
        taps: list[EmitConfig] = []
        for index, sliding_window_size in enumerate(sliding_window_sizes):
            path = ParameterPath("model.layers") / index
            attention_config = AttentionConfig(
                qkvg_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkvg_biases=self.attention_bias,
                has_out_biases=False,
                num_heads=self.num_attention_heads,
                num_groups=self.num_attention_heads,
                head_dim=head_dim,
                scale=None,
                is_causal=False,
                sliding_window_size=sliding_window_size,
                rope_config=global_rope_config if sliding_window_size is None else local_rope_config,
            )
            attention = at(path / "attn", attention_config, load=load_modern_bert_attention)
            # The first ModernBERT layer has no pre-attention normalization.
            if index == 0:
                attention_branch = residual(attention)
            else:
                attention_branch = residual(at(path / "attn_norm", transformer_norm_config), attention)
            tap = EmitConfig(f"hidden_states.{index}")
            steps += [
                attention_branch,
                residual(
                    at(path / "mlp_norm", transformer_norm_config),
                    at(path / "mlp", mlp_config, load=load_modern_bert_mlp),
                ),
                tap,
            ]
            taps.append(tap)
        steps.append(at("model.final_norm", transformer_norm_config))

        prediction_head_dense_config = LinearConfig()
        prediction_head_norm_config = NormalizationConfig(
            epsilon=self.norm_eps,
            scale_offset=0.0,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True,
        )
        prediction_head_activation = GELU(approximate=False)
        prediction_head_readout_config = LinearConfig()
        prediction_head_config = PredictionHeadConfig(
            dense_config=prediction_head_dense_config,
            activation=prediction_head_activation,
            normalization_config=prediction_head_norm_config,
            readout_config=prediction_head_readout_config,
            use_dense_bias=self.classifier_bias,
        )

        output_labels = [self.id2label[idx] for idx in range(len(self.id2label))]

        return ClassifierConfig(
            embedding_config=at("model.embeddings.tok_embeddings", embedding_config),
            program=SequenceConfig(tuple(steps)),
            prediction_head_config=prediction_head_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            num_labels=len(self.id2label),
            classifier_pooling=PoolingType(self.classifier_pooling),
            output_labels=tuple(output_labels),
            taps=tuple(taps),
        )
