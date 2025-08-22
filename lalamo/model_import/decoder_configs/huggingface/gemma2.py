from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    Activation,
    AttentionConfig,
    DecoderConfig,
    DecoderLayerConfig,
    FullPrecisionLinearConfig,
    MLPConfig,
    RMSNormConfig,
    TiedEmbeddingConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)

from .common import HuggingFaceConfig

__all__ = ["HFGemma2Config"]


@dataclass(frozen=True)
class HFGemma2Config(HuggingFaceConfig):
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

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        sliding_window_sizes = tuple(
            self.sliding_window if not bool(i % 2) else None for i in range(self.num_hidden_layers)
        )
        embedding_input_scale = self.hidden_size**0.5
        attention_scale = self.query_pre_attn_scalar**-0.5
        embedding_config = TiedEmbeddingConfig(
            input_scale=embedding_input_scale,
            logits_soft_cap=self.final_logit_softcapping,
            precision=activation_precision,
        )
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
        )
        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            logit_soft_cap=self.attn_logit_softcapping,
            has_qkv_biases=self.attention_bias,
            has_out_biases=False,
        )
        mlp_config = MLPConfig(
            linear_config=linear_config,
            activation=Activation.GELU,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=rmsnorm_config,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=rmsnorm_config,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.head_dim,
            attention_scale=attention_scale,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=sliding_window_sizes,
            context_length=context_length or self.max_position_embeddings,
        )
