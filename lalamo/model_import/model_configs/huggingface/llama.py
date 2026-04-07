from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DenseMLPConfig,
    EmbeddingQuantConfig,
    LinearConfig,
    LlamaRoPEConfig,
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
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        quantization = self.quantization or self.quantization_config
        if isinstance(quantization, MLXQuantizationConfig):
            quant_config = EmbeddingQuantConfig(
                group_size=quantization.group_size,
                bits=quantization.bits,
            )
            if self.tie_word_embeddings:
                embedding_config = TiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    quantization=quant_config,
                )
            else:
                embedding_config = UntiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    input_quantization=quant_config,
                    output_quantization=quant_config,
                )
        else:  # noqa: PLR5501
            if self.tie_word_embeddings:
                embedding_config = TiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                )
            else:
                embedding_config = UntiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                )
        if self.rope_scaling is None:
            rope_config = UnscaledRoPEConfig(
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
                head_dim=self.head_dim or self.hidden_size // self.num_attention_heads,
            )
        elif isinstance(self.rope_scaling, YarnRopeScalingConfig):
            rope_config = YARNRoPEConfig(
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
                original_context_length=self.rope_scaling.original_max_position_embeddings,
                beta_fast=self.rope_scaling.beta_fast,
                beta_slow=self.rope_scaling.beta_slow,
                truncate=self.rope_scaling.truncate,
                head_dim=self.head_dim or self.hidden_size // self.num_attention_heads,
            )
        elif isinstance(self.rope_scaling, LlamaRopeScalingConfig):
            rope_config = LlamaRoPEConfig(
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
                original_context_length=self.rope_scaling.original_max_position_embeddings,
                low_frequency_factor=self.rope_scaling.low_freq_factor,
                high_frequency_factor=self.rope_scaling.high_freq_factor,
                head_dim=self.head_dim or self.hidden_size // self.num_attention_heads,
            )
        else:
            raise ValueError("Unsupported rope_scaling configuration")
        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        linear_config = LinearConfig()
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
            head_dim=self.head_dim or self.hidden_size // self.num_attention_heads,
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
            rope_config=rope_config,
        )
        transformer_config = TransformerConfig(
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
