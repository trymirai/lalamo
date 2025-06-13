from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jaxtyping import DTypeLike

from lalamo.modules import (
    DecoderConfig,
    TiedEmbeddingConfig,
)
from lalamo.modules.activations import Activation
from lalamo.modules.attention import AttentionConfig
from lalamo.modules.decoder_layer import DecoderLayerConfig
from lalamo.modules.linear import FullPrecisionLinearConfig
from lalamo.modules.mlp import MLPConfig
from lalamo.modules.normalization import RMSNormConfig, UpcastMode
from lalamo.modules.rope import LinearScalingRoPEConfig, UnscaledRoPEConfig

from .common import HuggingFaceConfig

__all__ = ["HFGemma3Config", "HFGemma3TextConfig"]


NUM_SLIDING_WINDOW_LAYERS_PER_FULL_ATTENTION_LAYER = 6


def _round_to_bfloat16(x: float) -> float:
    return jnp.asarray(x).astype(jnp.bfloat16).item()


@dataclass(frozen=True)
class GemmaRoPEScalingConfig:
    factor: float
    rope_type: Literal["linear"]


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
    rope_theta: float = 1000000.0
    rope_local_base_freq: float = 10000.0
    rope_scaling: GemmaRoPEScalingConfig | None = None
    final_logit_softcapping: float | None = None
    vocab_size: int = 262208

    @property
    def sliding_window_sizes(self) -> list[int | None]:
        result = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % NUM_SLIDING_WINDOW_LAYERS_PER_FULL_ATTENTION_LAYER == 0:
                result.append(None)
            else:
                result.append(self.sliding_window)
        return result

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        input_scale = _round_to_bfloat16(self.hidden_size**0.5)
        attention_scale = self.query_pre_attn_scalar**-0.5
        embedding_config = TiedEmbeddingConfig(
            input_scale=input_scale,
            logits_soft_cap=None,
            precision=activation_precision,
        )
        rms_norm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
        )

        if self.rope_scaling is not None:
            global_rope_config = LinearScalingRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
            )
        else:
            global_rope_config = UnscaledRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=self.max_position_embeddings,
            )
        local_rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_local_base_freq,
            max_sequence_length=self.max_position_embeddings,
        )

        linear_config = FullPrecisionLinearConfig(precision=activation_precision)
        mlp_config = MLPConfig(linear_config=linear_config, activation=Activation.GELU)
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=rms_norm_config,
            key_norm_config=rms_norm_config,
            logit_soft_cap=self.attn_logit_softcapping,
            has_qkv_biases=self.attention_bias,
            has_out_biases=self.attention_bias,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rms_norm_config,
            attention_config=attention_config,
            post_attention_norm_config=rms_norm_config,
            pre_mlp_norm_config=rms_norm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=rms_norm_config,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            global_rope_config=global_rope_config,
            local_rope_config=local_rope_config,
            layer_config=decoder_layer_config,
            output_norm_config=rms_norm_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.head_dim,
            attention_scale=attention_scale,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=tuple(self.sliding_window_sizes),
            context_length=context_length or self.max_position_embeddings,
        )


@dataclass(frozen=True)
class HFGemma3TextConfig(HFGemma3TextConfigRaw, HuggingFaceConfig):
    pass


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
class HFGemma3Config(HuggingFaceConfig):
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

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        return self.text_config.to_decoder_config(
            context_length=context_length,
            activation_precision=activation_precision,
            accumulation_precision=accumulation_precision,
        )
