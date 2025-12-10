from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    LlamaRoPEConfig,
    MLXQuantizedLinearConfig,
    MLXQuantizedTiedEmbeddingConfig,
    MLXQuantizedUntiedEmbeddingConfig,
    NormalizationConfig,
    SiLU,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
    YARNRoPEConfig,
)
from lalamo.quantization import QuantizationMode

from .common import HuggingFaceLMConfig, MLXQuantizationConfig, QuantizationConfigType

__all__ = ["HFLlamaConfig"]


@dataclass(frozen=True)
class LlamaRopeScalingConfig:
    factor: float
    high_freq_factor: float
    low_freq_factor: float
    original_max_position_embeddings: int
    rope_type: Literal["llama3"]


@dataclass(frozen=True)
class YarnRopeScalingConfig:
    factor: float
    beta_fast: float
    beta_slow: float
    original_max_position_embeddings: int
    rope_type: Literal["yarn"]
    truncate: bool


@dataclass(frozen=True)
class HFLlamaConfig(HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[Literal["LlamaForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["llama"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_scaling: LlamaRopeScalingConfig | YarnRopeScalingConfig | None
    rope_theta: float
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    vocab_size: int
    head_dim: int | None = None

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        quantization = self.quantization or self.quantization_config
        if isinstance(quantization, MLXQuantizationConfig):
            if self.tie_word_embeddings:
                embedding_config = MLXQuantizedTiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    group_size=quantization.group_size,
                    embedding_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                    activation_quantization_mode=None,
                    activation_precision=activation_precision,
                )
            else:
                embedding_config = MLXQuantizedUntiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    group_size=quantization.group_size,
                    embedding_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                    activation_quantization_mode=None,
                    activation_precision=activation_precision,
                )
        else: # noqa: PLR5501
            if self.tie_word_embeddings:
                embedding_config = TiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    precision=activation_precision,
                )
            else:
                embedding_config = UntiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    precision=activation_precision,
                )
        if self.rope_scaling is None:
            rope_config = UnscaledRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
            )
        elif isinstance(self.rope_scaling, YarnRopeScalingConfig):
            rope_config = YARNRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
                original_context_length=self.rope_scaling.original_max_position_embeddings,
                beta_fast=self.rope_scaling.beta_fast,
                beta_slow=self.rope_scaling.beta_slow,
                truncate=self.rope_scaling.truncate,
            )
        elif isinstance(self.rope_scaling, LlamaRopeScalingConfig):
            rope_config = LlamaRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
                original_context_length=self.rope_scaling.original_max_position_embeddings,
                low_frequency_factor=self.rope_scaling.low_freq_factor,
                high_frequency_factor=self.rope_scaling.high_freq_factor,
            )
        else:
            raise ValueError("Unsupported rope_scaling configuration")
        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        if quantization is None:
            linear_config = FullPrecisionLinearConfig(
                precision=activation_precision,
            )
        elif isinstance(quantization, MLXQuantizationConfig):
            linear_config = MLXQuantizedLinearConfig(
                group_size=quantization.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )
        else:
            linear_config = GroupQuantizedLinearConfig(
                group_size=quantization.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
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
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=(self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads),
            is_causal=True,
            scale=None,
            sliding_window_size=None,
        )
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )
        transformer_layer_config = TransformerLayerConfig(
            pre_mixer_norm_config=rmsnorm_config,
            mixer_config=attention_config,
            post_mixer_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )
        transformer_config = TransformerConfig(
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_configs=(transformer_layer_config,) * self.num_hidden_layers,
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=context_length or self.max_position_embeddings,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
