from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig, UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import YARNRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig

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
    vocab_size: int
    head_dim: int

    # Bonsai's config.json doesn't include torch_dtype
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    quantization: QuantizationConfigType = None

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        assert isinstance(self.quantization, MLXQuantizationConfig), "HFBonsaiConfig requires MLX quantization config"
        assert not self.use_sliding_window, "Sliding window attention is not supported for Bonsai"
        max_sequence_length = self.max_position_embeddings if context_length is None else context_length
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

        rope_config = YARNRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=max_sequence_length,
            head_dim=self.head_dim,
            scaling_factor=self.rope_scaling.factor,
            original_context_length=self.rope_scaling.original_max_position_embeddings,
            beta_fast=self.rope_scaling.beta_fast,
            beta_slow=self.rope_scaling.beta_slow,
            truncate=True,
        )

        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
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
            rope_config=rope_config,
        )
        transformer_config = TransformerConfig(
            layer_configs=(transformer_layer_config,) * self.num_hidden_layers,
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
