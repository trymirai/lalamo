from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DenseMLPConfig,
    MLXQuantizedLinearConfig,
    MLXQuantizedTiedEmbeddingConfig,
    MLXQuantizedUntiedEmbeddingConfig,
    NormalizationConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UpcastMode,
    YARNRoPEConfig,
)
from lalamo.modules.activations import SiLU
from lalamo.quantization import QuantizationMode

from .common import HuggingFaceLMConfig, MLXQuantizationConfig, QuantizationConfigType

__all__ = ["HFBonsaiConfig"]


@dataclass(frozen=True)
class BonsaiYarnRopeScalingConfig:
    rope_type: Literal["yarn"]
    factor: float
    original_max_position_embeddings: int
    # HuggingFace YARN defaults, not present in Bonsai's config.json:
    # https://github.com/huggingface/transformers/blob/6abd9725ee7d809dc974991f8ff6c958afb63a3a/src/transformers/modeling_rope_utils.py#L591
    beta_fast: float = 32.0
    beta_slow: float = 1.0


@dataclass(frozen=True)
class HFBonsaiConfig(HuggingFaceLMConfig):
    eos_token_id: int | list[int]
    attention_bias: bool
    hidden_act: Literal["silu"]
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    model_type: Literal["qwen3"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: BonsaiYarnRopeScalingConfig
    tie_word_embeddings: bool
    use_sliding_window: bool
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    vocab_size: int
    head_dim: int

    quantization: QuantizationConfigType = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        assert isinstance(self.quantization, MLXQuantizationConfig), "HFBonsaiConfig requires MLX quantization config"
        assert not self.use_sliding_window, "Sliding window attention is not supported for Bonsai"
        quantization = self.quantization
        quantization_mode = QuantizationMode.from_num_bits(quantization.bits)

        if self.tie_word_embeddings:
            embedding_config = MLXQuantizedTiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
                group_size=quantization.group_size,
                embedding_quantization_mode=quantization_mode,
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )
        else:
            embedding_config = MLXQuantizedUntiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
                group_size=quantization.group_size,
                embedding_quantization_mode=quantization_mode,
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )

        rope_config = YARNRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=context_length or self.max_position_embeddings,
            scaling_factor=self.rope_scaling.factor,
            original_context_length=self.rope_scaling.original_max_position_embeddings,
            beta_fast=self.rope_scaling.beta_fast,
            beta_slow=self.rope_scaling.beta_slow,
            truncate=True,
        )

        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        linear_config = MLXQuantizedLinearConfig(
            group_size=quantization.group_size,
            weight_quantization_mode=quantization_mode,
            activation_quantization_mode=None,
            activation_precision=activation_precision,
        )
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        layer_configs = []
        for _ in range(self.num_hidden_layers):
            attention_config = AttentionConfig(
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
            )
            transformer_layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=rmsnorm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=None,
                pre_mlp_norm_config=rmsnorm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
            )
            layer_configs.append(transformer_layer_config)
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
