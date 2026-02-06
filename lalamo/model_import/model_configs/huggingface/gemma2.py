from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    NormalizationConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.activations import GELU

from .common import HuggingFaceLMConfig

__all__ = ["HFGemma2Config"]


@dataclass(frozen=True)
class HFGemma2Config(HuggingFaceLMConfig):
    architectures: list[Literal["Gemma2ForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    attn_logit_softcapping: float
    bos_token_id: int | list[int]
    cache_implementation: Literal["hybrid"]
    eos_token_id: int | list[int]
    final_logit_softcapping: float
    head_dim: int
    hidden_act: Literal["gelu_pytorch_tanh"]
    hidden_activation: Literal["gelu_pytorch_tanh"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: Literal["gemma2"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    query_pre_attn_scalar: float
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    transformers_version: str
    use_cache: bool
    vocab_size: int
    torch_dtype: Literal["bfloat16", "float16", "float32"]

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        embedding_input_scale = self.hidden_size**0.5
        attention_scale = self.query_pre_attn_scalar**-0.5
        embedding_config = TiedEmbeddingConfig(
            input_scale=embedding_input_scale,
            logit_soft_cap=self.final_logit_softcapping,
            precision=activation_precision,
        )
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
        )
        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=GELU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        layer_configs = []
        for i in range(self.num_hidden_layers):
            sliding_window_size = self.sliding_window if not bool(i % 2) else None
            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=self.attn_logit_softcapping,
                has_sinks=False,
                has_qkv_biases=self.attention_bias,
                has_out_biases=False,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=True,
                scale=attention_scale,
                sliding_window_size=sliding_window_size,
            )
            transformer_layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=rmsnorm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=rmsnorm_config,
                pre_mlp_norm_config=rmsnorm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=rmsnorm_config,
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
