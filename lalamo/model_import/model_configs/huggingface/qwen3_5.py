import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from jaxtyping import Array, DTypeLike

from lalamo.model_import.loaders import load_huggingface_decoder
from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DeltaNetAttentionConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    MLXQuantizedTiedEmbeddingConfig,
    MLXQuantizedUntiedEmbeddingConfig,
    NormalizationConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
)
from lalamo.modules.activations import SiLU
from lalamo.modules.common import LalamoModule
from lalamo.modules.decoder import Decoder
from lalamo.modules.linear import MLXQuantizedLinearConfig
from lalamo.modules.token_mixers import SeparableCausalConvConfig
from lalamo.quantization import QuantizationMode

from .common import HuggingFaceLMConfig, MLXQuantizationConfig, QuantizationConfigType

__all__ = ["HFQwen35Config", "HFQwen35TextConfig"]


def _layer_types_from_fields(
    layer_types: list[Literal["linear_attention", "full_attention"]] | None,
    full_attention_interval: int | None,
    num_hidden_layers: int,
) -> list[Literal["linear_attention", "full_attention"]]:
    if layer_types is not None:
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"Expected {num_hidden_layers} layer types, got {len(layer_types)}.",
            )
        return layer_types

    if full_attention_interval is None or full_attention_interval <= 0:
        raise ValueError("Qwen3.5 config requires `layer_types` or positive `full_attention_interval`.")

    return [
        ("full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention")
        for i in range(num_hidden_layers)
    ]


@dataclass(frozen=True)
class HFQwen35TextConfigRaw:
    model_type: Literal["qwen3_5_text"]
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    vocab_size: int
    rope_parameters: dict[str, Any]

    head_dim: int
    attention_bias: bool
    layer_types: list[Literal["linear_attention", "full_attention"]]
    full_attention_interval: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int

    quantization: QuantizationConfigType | None = None
    quantization_config: QuantizationConfigType | None = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
        fallback_quantization: QuantizationConfigType | None = None,
    ) -> DecoderConfig:
        if self.quantization is not None:
            quantization = self.quantization
        elif self.quantization_config is not None:
            quantization = self.quantization_config
        else:
            quantization = fallback_quantization

        is_mlx = isinstance(quantization, MLXQuantizationConfig)

        if is_mlx:
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
        else:  # noqa: PLR5501
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

        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_parameters["rope_theta"],
            max_sequence_length=context_length or self.max_position_embeddings,
        )

        # Qwen3.5 RMSNorm computes (1 + weight) * norm(x). HF stores raw weights,
        # but MLX community converters bake the +1 into the weights, so no offset needed.
        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None if is_mlx else 1.0,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        # Qwen3.5 DeltaNet RMSNorm uses direct weights (no +1 offset).
        gated_rmsnorm_config = NormalizationConfig(
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

        partial_rotary_factor = self.rope_parameters["partial_rotary_factor"]
        resolved_layer_types = _layer_types_from_fields(
            self.layer_types,
            self.full_attention_interval,
            self.num_hidden_layers,
        )

        layer_configs = []
        for layer_type in resolved_layer_types:
            if layer_type == "linear_attention":
                mixer_config = DeltaNetAttentionConfig(
                    in_proj_config=linear_config,
                    conv_config=SeparableCausalConvConfig(
                        precision=activation_precision,
                        has_biases=False,
                    ),
                    out_proj_config=linear_config,
                    norm_config=gated_rmsnorm_config,
                    num_heads=self.linear_num_value_heads,
                    num_groups=self.linear_num_key_heads,
                    head_dim=self.linear_key_head_dim,
                    value_head_dim=self.linear_value_head_dim,
                    kernel_size=self.linear_conv_kernel_dim,
                )
            else:
                mixer_config = AttentionConfig(
                    qkv_projection_config=linear_config,
                    out_projection_config=linear_config,
                    query_norm_config=rmsnorm_config,
                    key_norm_config=rmsnorm_config,
                    logit_soft_cap=None,
                    has_sinks=False,
                    has_qkv_biases=self.attention_bias,
                    has_out_biases=self.attention_bias,
                    num_heads=self.num_attention_heads,
                    num_groups=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    is_causal=True,
                    scale=None,
                    sliding_window_size=None,
                    has_gate=True,
                    partial_rope_dim=int(self.head_dim * partial_rotary_factor),
                )

            layer_configs.append(
                TransformerLayerConfig(
                    pre_mixer_norm_config=rmsnorm_config,
                    mixer_config=mixer_config,
                    post_mixer_norm_config=None,
                    pre_mlp_norm_config=rmsnorm_config,
                    mlp_config=DenseMLPConfig(
                        linear_config=linear_config,
                        activation=SiLU(),
                        has_up_biases=False,
                        has_down_biases=False,
                        up_clipping=None,
                        gate_clipping=None,
                    ),
                    post_mlp_norm_config=None,
                ),
            )

        transformer_config = TransformerConfig(
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_configs=tuple(layer_configs),
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


@dataclass(frozen=True)
class HFQwen35TextConfig(HFQwen35TextConfigRaw, HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    eos_token_id: int | list[int] = field(default_factory=list)


@dataclass(frozen=True)
class HFQwen35Config(HuggingFaceLMConfig):
    model_type: Literal["qwen3_5"]
    tie_word_embeddings: bool
    text_config: HFQwen35TextConfigRaw
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    eos_token_id: int | list[int] = field(default_factory=list)
    quantization: QuantizationConfigType | None = None
    quantization_config: QuantizationConfigType | None = None

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        # tie_word_embeddings lives at the top level but text_config needs it
        if "tie_word_embeddings" in config and "text_config" in config:
            config["text_config"].setdefault("tie_word_embeddings", config["tie_word_embeddings"])
        return cls._converter.structure(config, cls)

    def _load_weights(
        self,
        model: LalamoModule,
        weights_dict: Mapping[str, Array],
    ) -> LalamoModule:
        new_weights = {k.replace("model.language_model.", "model.", 1): v for k, v in weights_dict.items()}

        assert isinstance(model, Decoder)
        return load_huggingface_decoder(model, new_weights)

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
