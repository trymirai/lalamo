import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Self

import equinox as eqx
from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders.vision import load_qwen3_5_vision_encoder
from lalamo.model_import.model_configs.foreign_config import ForeignVisionLanguageConfig
from lalamo.models import GenerationConfig, VisionLanguageModel, VisionLanguageModelConfig
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig, UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, MixtureOfExpertsConfig, SoftmaxRouting
from lalamo.modules.mrope import MRoPEConfig, MRoPELayout
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNetConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig
from lalamo.modules.vision import MRoPEImagePositions
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig, MLXQuantizationConfig, QuantizationConfigType
from .qwen3_5_vision import HFQwen35ImageProcessorConfig, HFQwen35VisionConfig

__all__ = ["HFQwen35Config"]


@dataclass(frozen=True)
class Qwen35RopeParameters:
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int] | None = None
    mrope_interleaved: bool = False


@dataclass(frozen=True)
class HFQwen35Config(HuggingFaceLMConfig, ForeignVisionLanguageConfig):
    model_type: Literal["qwen3_5", "qwen3_5_moe"]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    vocab_size: int
    rope_parameters: Qwen35RopeParameters

    head_dim: int
    attention_bias: bool
    layer_types: list[Literal["linear_attention", "full_attention"]]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int

    intermediate_size: int | None = None

    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None

    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    eos_token_id: int | list[int] = field(default_factory=list)
    quantization: QuantizationConfigType | None = None

    # Present in the multimodal checkpoints; ignored unless vision is requested at import time.
    vision_config: HFQwen35VisionConfig | None = None
    image_token_id: int | None = None

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        # Multimodal configs nest text model params under text_config;
        # top-level keys override text_config on conflict.
        text_config = config.pop("text_config", {})
        merged = {**text_config, **config}
        return cls._converter.structure(merged, cls)

    def to_vision_language_config(
        self,
        metadata_dict: Mapping[str, str],
        token_codec_config: ChatCodecConfig,
        generation_config: GenerationConfig,
        image_processor_config_path: Path,
    ) -> VisionLanguageModelConfig:
        if self.vision_config is None or self.image_token_id is None:
            raise ValueError("This Qwen3.5 checkpoint has no vision tower.")
        mrope_section = self.rope_parameters.mrope_section
        if mrope_section is None or not self.rope_parameters.mrope_interleaved:
            raise ValueError("Qwen3.5 vision requires interleaved M-RoPE parameters in the config.")

        decoder_config = self.to_decoder_config(metadata_dict)
        # M-RoPE equals the plain RoPE on text, so only the rope configs change; weights are identical.
        layer_configs = tuple(
            replace(
                layer_config,
                rope_config=MRoPEConfig(
                    base=layer_config.rope_config.base,
                    max_sequence_length=layer_config.rope_config.max_sequence_length,
                    head_dim=layer_config.rope_config.head_dim,
                    sections=mrope_section,
                    layout=MRoPELayout.INTERLEAVED,
                ),
            )
            if layer_config.rope_config is not None
            else layer_config
            for layer_config in decoder_config.transformer_config.layer_configs
        )
        decoder_config = replace(
            decoder_config,
            transformer_config=replace(decoder_config.transformer_config, layer_configs=layer_configs),
        )
        return VisionLanguageModelConfig(
            token_codec_config=token_codec_config,
            decoder_config=decoder_config,
            generation_config=generation_config,
            vision_encoder_config=self.vision_config.to_vision_encoder_config(),
            image_preprocessing_config=HFQwen35ImageProcessorConfig.from_json(
                image_processor_config_path,
            ).to_image_preprocessing_config(),
            image_position_scheme=MRoPEImagePositions(),
            image_token_id=self.image_token_id,
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        model = super()._load_weights(model, weights_dict, implementation=implementation)
        if not isinstance(model, VisionLanguageModel):
            return model
        vision_path = ParameterPath("model.visual")
        if not any(key.startswith(f"{vision_path}.") for key in weights_dict):
            raise ValueError(f"The checkpoint has no vision weights under {vision_path}.")
        vision_encoder = load_qwen3_5_vision_encoder(
            model.vision_encoder,
            weights_dict,
            vision_path,
            implementation=implementation,
        )
        return eqx.tree_at(lambda m: m.vision_encoder, model, vision_encoder)

    def to_decoder_config(
        self,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
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
        is_mlx = isinstance(self.quantization, MLXQuantizationConfig)

        rope_config = UnscaledRoPEConfig(
            base=self.rope_parameters.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=int(self.head_dim * self.rope_parameters.partial_rotary_factor),
        )

        # Qwen3.5 RMSNorm computes (1 + weight) * norm(x). HF stores raw weights,
        # but MLX community converters bake the +1 into the weights, so no offset needed.
        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None if is_mlx else 1.0,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        # Qwen3.5 DeltaNet RMSNorm uses direct weights (no +1 offset).
        gated_rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        linear_config = LinearConfig()

        if self.model_type == "qwen3_5_moe":
            assert self.num_experts is not None
            assert self.num_experts_per_tok is not None
            assert self.moe_intermediate_size is not None
            experts_config = DenseMLPConfig(
                linear_config=linear_config,
                activation=SiLU(),
                has_up_biases=False,
                has_down_biases=False,
                up_clipping=None,
                gate_clipping=None,
            )
            mlp_config = MixtureOfExpertsConfig(
                num_routed_experts=self.num_experts,
                num_active_routed_experts=self.num_experts_per_tok,
                routing_function=SoftmaxRouting(),
                router_config=linear_config,
                router_has_biases=False,
                expert_config=experts_config,
                gate_config=linear_config,
                num_shared_experts=1,
                expert_hidden_dim=self.moe_intermediate_size,
            )
            hidden_dim = self.moe_intermediate_size
        else:
            assert self.intermediate_size is not None
            mlp_config = DenseMLPConfig(
                linear_config=linear_config,
                activation=SiLU(),
                has_up_biases=False,
                has_down_biases=False,
                up_clipping=None,
                gate_clipping=None,
            )
            hidden_dim = self.intermediate_size

        layer_configs = []
        for layer_type in self.layer_types:
            if layer_type == "linear_attention":
                mixer_config = DeltaNetConfig(
                    in_proj_config=linear_config,
                    conv_config=SeparableCausalConvConfig(
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
                    qkvg_projection_config=linear_config,
                    out_projection_config=linear_config,
                    query_norm_config=rmsnorm_config,
                    key_norm_config=rmsnorm_config,
                    logit_soft_cap=None,
                    has_sinks=False,
                    has_qkvg_biases=self.attention_bias,
                    has_out_biases=self.attention_bias,
                    num_heads=self.num_attention_heads,
                    num_groups=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    is_causal=True,
                    scale=None,
                    sliding_window_size=None,
                    has_gate=True,
                )

            layer_configs.append(
                TransformerLayerConfig(
                    pre_mixer_norm_config=rmsnorm_config,
                    mixer_config=mixer_config,
                    post_mixer_norm_config=None,
                    pre_mlp_norm_config=rmsnorm_config,
                    mlp_config=mlp_config,
                    post_mlp_norm_config=None,
                    rope_config=rope_config if layer_type != "linear_attention" else None,
                ),
            )

        transformer_config = TransformerConfig(
            layer_configs=tuple(layer_configs),
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=hidden_dim,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
