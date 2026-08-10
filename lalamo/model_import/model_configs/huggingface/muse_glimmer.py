from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig

from .common import HuggingFaceLMConfig, QuantizationConfigType

__all__ = ["HFMuseGlimmerConfig"]


@dataclass(frozen=True)
class MuseGlimmerRopeParameters:
    rope_theta: float
    rope_type: Literal["default"]


@dataclass(frozen=True)
class HFMuseGlimmerVisionConfig:
    hidden_act: Literal["gelu"]
    hidden_size: int
    intermediate_size: int
    layer_norm_eps: float
    layer_types: list[Literal["window_attention", "full_attention"]]
    max_position_embeddings: int
    merge_size: int
    model_type: Literal["muse_glimmer_vision"]
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    patch_temporal: int
    pos_emb_height: int
    pos_emb_width: int
    rope_parameters: MuseGlimmerRopeParameters


@dataclass(frozen=True)
class HFMuseGlimmerTextConfig:
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    final_logit_softcapping: float | None
    head_dim: int
    hidden_activation: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    layer_rope_theta: list[float]
    layer_types: list[Literal["sliding_attention", "full_attention"]]
    max_position_embeddings: int
    model_type: Literal["muse_glimmer_text"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    output_multiplier: float
    pad_token_id: int | None
    post_norm_eps: float
    qk_scale_factor: float
    rms_norm_eps: float
    rope_parameters: MuseGlimmerRopeParameters
    sliding_window: int
    tie_word_embeddings: bool
    use_cache: bool
    vocab_size: int

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        max_sequence_length = self.max_position_embeddings if context_length is None else context_length

        embedding_config = UntiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=self.final_logit_softcapping,
            logit_scale=self.output_multiplier,
        )
        # Muse Glimmer applies a weightless RMS norm on top of the embeddings.
        embedding_norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
            has_scale=False,
        )

        pre_norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )
        post_norm_config = NormalizationConfig(
            epsilon=self.post_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )
        output_norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )
        # The query/key norm is weightless.
        qk_norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
            has_scale=False,
        )

        rope_config = UnscaledRoPEConfig(
            base=self.rope_parameters.rope_theta,
            max_sequence_length=max_sequence_length,
            head_dim=self.head_dim,
        )

        linear_config = LinearConfig()
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        attention_scale = self.qk_scale_factor * self.head_dim**-0.5

        layer_configs = []
        for layer_idx, layer_type in enumerate(self.layer_types):
            is_sliding = layer_type == "sliding_attention"
            # Full-attention layers are NoPE (`layer_rope_theta == 0`); sliding layers use RoPE.
            if is_sliding:
                assert self.layer_rope_theta[layer_idx] == self.rope_parameters.rope_theta
                layer_rope_config: UnscaledRoPEConfig | None = rope_config
                sliding_window_size: int | None = self.sliding_window
            else:
                assert self.layer_rope_theta[layer_idx] == 0
                layer_rope_config = None
                sliding_window_size = None

            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=qk_norm_config,
                key_norm_config=qk_norm_config,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=self.attention_bias,
                has_out_biases=self.attention_bias,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=True,
                scale=attention_scale,
                sliding_window_size=sliding_window_size,
                gate_projection_config=linear_config,
            )
            transformer_layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=pre_norm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=post_norm_config,
                pre_mlp_norm_config=pre_norm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=post_norm_config,
                rope_config=layer_rope_config,
            )
            layer_configs.append(transformer_layer_config)

        transformer_config = TransformerConfig(
            layer_configs=tuple(layer_configs),
            output_norm_config=output_norm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
            embedding_norm_config=embedding_norm_config,
        )


@dataclass(frozen=True)
class HFMuseGlimmerConfig(HuggingFaceLMConfig):
    architectures: list[Literal["MuseGlimmerForConditionalGeneration"]]
    dtype: Literal["bfloat16", "float16", "float32"]
    image_token_id: int
    model_type: Literal["muse_glimmer"]
    out_hidden_size: int
    projector_hidden_act: Literal["gelu"]
    projector_hidden_size: int
    text_config: HFMuseGlimmerTextConfig
    transformers_version: str
    video_token_id: int
    vision_config: HFMuseGlimmerVisionConfig

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    @property
    def eos_token_ids(self) -> list[int]:
        return [self.text_config.eos_token_id]

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig:
        return self.text_config.to_decoder_config(
            context_length=context_length,
            metadata_dict=metadata_dict,
        )
