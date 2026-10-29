from dataclasses import dataclass, field
from typing import Literal

from lalamo.model_import.loaders.binding import Binder
from lalamo.modules.activations import GELU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import LinearScalingRoPEConfig, UnscaledRoPEConfig, YARNRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig

from .common import HuggingFaceLMConfig, QuantizationConfigType, residual_decoder

__all__ = ["HFGemma3Config", "HFGemma3TextConfig"]


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

    def build_decoder(self, at: Binder) -> DecoderConfig:
        input_scale = self.hidden_size**0.5
        attention_scale = self.query_pre_attn_scalar**-0.5
        embedding_config = TiedEmbeddingConfig(
            input_scale=input_scale,
            logit_soft_cap=self.final_logit_softcapping,
        )
        rms_norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )

        if isinstance(self.rope_scaling, GemmaRoPEScalingConfig):
            global_rope_config = LinearScalingRoPEConfig(
                base=self.rope_theta,
                max_sequence_length=self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
                head_dim=self.head_dim,
            )
        elif isinstance(self.rope_scaling, YarnRopeScalingConfig):
            global_rope_config = YARNRoPEConfig(
                base=self.rope_theta,
                scaling_factor=self.rope_scaling.factor,
                max_sequence_length=self.max_position_embeddings,
                original_context_length=self.rope_scaling.original_max_position_embeddings,
                beta_fast=self.rope_scaling.beta_fast,
                beta_slow=self.rope_scaling.beta_slow,
                truncate=self.rope_scaling.truncate,
                head_dim=self.head_dim,
            )
        elif self.rope_scaling is None:
            global_rope_config = UnscaledRoPEConfig(
                base=self.rope_theta,
                max_sequence_length=self.max_position_embeddings,
                head_dim=self.head_dim,
            )
        else:
            raise ValueError("Invalid rope scaling configuration")

        local_rope_config = UnscaledRoPEConfig(
            base=self.rope_local_base_freq,
            max_sequence_length=self.max_position_embeddings,
            head_dim=self.head_dim,
        )

        linear_config = LinearConfig()
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=GELU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
            hidden_dim=self.intermediate_size,
        )
        # Global layers (no sliding window) use the global RoPE, local layers the local one.
        attention_configs = [
            AttentionConfig(
                qkvg_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=rms_norm_config,
                key_norm_config=rms_norm_config,
                logit_soft_cap=self.attn_logit_softcapping,
                has_sinks=False,
                has_qkvg_biases=self.attention_bias,
                has_out_biases=self.attention_bias,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=True,
                scale=attention_scale,
                sliding_window_size=sliding_window_size,
                rope_config=global_rope_config if sliding_window_size is None else local_rope_config,
            )
            for sliding_window_size in self.sliding_window_sizes
        ]
        return residual_decoder(
            at,
            embedding=at("model.embed_tokens", embedding_config),
            layers=[
                (at(f"model.layers.{index}.self_attn", attention), at(f"model.layers.{index}.mlp", mlp_config))
                for index, attention in enumerate(attention_configs)
            ],
            norm=rms_norm_config,
            model_dim=self.hidden_size,
            vocab_size=self.vocab_size,
            mlp_norm="pre_feedforward_layernorm",
            post_mixer_norm="post_attention_layernorm",
            post_mlp_norm="post_feedforward_layernorm",
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

    def build_decoder(self, at: Binder) -> DecoderConfig:
        return self.text_config.build_decoder(at)
