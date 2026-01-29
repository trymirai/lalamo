from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
from jaxtyping import DTypeLike

from lalamo.modules import (
    DecoderConfig,
    MLXQuantizedLinearConfig,
    MLXQuantizedTiedEmbeddingConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
)
from lalamo.modules.activations import GELU
from lalamo.modules.linear import FullPrecisionLinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import LinearScalingRoPEConfig, UnscaledRoPEConfig, YARNRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig
from lalamo.quantization import QuantizationMode

from .common import HuggingFaceLMConfig, MLXQuantizationConfig, QuantizationConfigType

__all__ = ["HFGemma3Config", "HFGemma3TextConfig"]


def _round_to_bfloat16(x: float) -> float:
    return jnp.asarray(x).astype(jnp.bfloat16).item()


@dataclass(frozen=True)
class GemmaRoPEScalingConfig:
    factor: float
    rope_type: Literal["linear"]


@dataclass(frozen=True)
class YarnRopeScalingConfig:
    factor: float
    beta_fast: float
    beta_slow: float
    original_max_position_embeddings: int
    rope_type: Literal["yarn"]
    truncate: bool = False


@dataclass(frozen=True)
class HFGemma3TextConfigRaw:
    hidden_size: int
    intermediate_size: int
    model_type: Literal["gemma3_text"]
    num_hidden_layers: int
    sliding_window: int
    rms_norm_eps: float = 1e-06
    query_pre_attn_scalar: float = 256.0
    attention_bias: bool = False
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    attn_logit_softcapping: float | None = None
    head_dim: int = 256
    max_position_embeddings: int = 131072
    sliding_window_pattern: int = 6
    rope_theta: float = 1000000.0
    rope_local_base_freq: float = 10000.0
    rope_scaling: GemmaRoPEScalingConfig | YarnRopeScalingConfig | None = None
    final_logit_softcapping: float | None = None
    vocab_size: int = 262208

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    @property
    def sliding_window_sizes(self) -> list[int | None]:
        result = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.sliding_window_pattern == 0:
                result.append(None)
            else:
                result.append(self.sliding_window)
        return result

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
        fallback_quantization: QuantizationConfigType | None = None,
    ) -> DecoderConfig:
        quantization = self.quantization or self.quantization_config or fallback_quantization
        input_scale = _round_to_bfloat16(self.hidden_size**0.5)
        attention_scale = self.query_pre_attn_scalar**-0.5
        if quantization is None:
            embedding_config = TiedEmbeddingConfig(
                input_scale=input_scale,
                logit_soft_cap=self.final_logit_softcapping,
                precision=activation_precision,
            )
        elif isinstance(quantization, MLXQuantizationConfig):
            embedding_config = MLXQuantizedTiedEmbeddingConfig(
                input_scale=input_scale,
                logit_soft_cap=self.final_logit_softcapping,
                group_size=quantization.group_size,
                embedding_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )
        else:
            raise RuntimeError(f"Unsupported quantization format: {type(quantization)}")
        rms_norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )

        if isinstance(self.rope_scaling, GemmaRoPEScalingConfig):
            global_rope_config = LinearScalingRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
            )
        elif isinstance(self.rope_scaling, YarnRopeScalingConfig):
            global_rope_config = YARNRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                scaling_factor=self.rope_scaling.factor,
                max_sequence_length=self.max_position_embeddings,
                original_context_length=self.rope_scaling.original_max_position_embeddings,
                beta_fast=self.rope_scaling.beta_fast,
                beta_slow=self.rope_scaling.beta_slow,
                truncate=self.rope_scaling.truncate,
            )
        elif self.rope_scaling is None:
            global_rope_config = UnscaledRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
            )
        else:
            raise ValueError("Invalid rope scaling configuration")

        local_rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_local_base_freq,
            max_sequence_length=context_length or self.max_position_embeddings,
        )

        if quantization is None:
            linear_config = FullPrecisionLinearConfig(precision=activation_precision)
        elif isinstance(quantization, MLXQuantizationConfig):
            linear_config = MLXQuantizedLinearConfig(
                group_size=quantization.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )
        else:
            raise RuntimeError(f"Unsupported quantization format: {type(quantization)}")
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=GELU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )
        layer_configs = []
        for sliding_window_size in self.sliding_window_sizes:
            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=rms_norm_config,
                key_norm_config=rms_norm_config,
                logit_soft_cap=self.attn_logit_softcapping,
                has_sinks=False,
                has_qkv_biases=self.attention_bias,
                has_out_biases=self.attention_bias,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=True,
                scale=attention_scale,
                sliding_window_size=sliding_window_size,
            )
            transformer_layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=rms_norm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=rms_norm_config,
                pre_mlp_norm_config=rms_norm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=rms_norm_config,
            )
            layer_configs.append(transformer_layer_config)

        transformer_config = TransformerConfig(
            global_rope_config=global_rope_config,
            local_rope_config=local_rope_config,
            layer_configs=tuple(layer_configs),
            output_norm_config=rms_norm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=context_length or self.max_position_embeddings,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )


@dataclass(frozen=True)
class HFGemma3TextConfig(HFGemma3TextConfigRaw, HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    eos_token_id: int | list[int] = field(default_factory=list)


@dataclass(frozen=True)
class HFGemma3VisionConfig:
    hidden_size: int
    image_size: int
    intermediate_size: int
    model_type: Literal["siglip_vision_model"]
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    vision_use_head: bool


@dataclass(frozen=True)
class HFGemma3Config(HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[Literal["Gemma3ForConditionalGeneration"]]
    boi_token_index: int
    eoi_token_index: int
    eos_token_id: int | list[int]
    image_token_index: int
    initializer_range: float
    mm_tokens_per_image: int
    model_type: Literal["gemma3"]
    text_config: HFGemma3TextConfigRaw
    transformers_version: str
    vision_config: HFGemma3VisionConfig

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig:
        quantization = self.quantization or self.quantization_config
        return self.text_config.to_decoder_config(
            context_length=context_length,
            activation_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            metadata_dict=metadata_dict,
            fallback_quantization=quantization,
        )
