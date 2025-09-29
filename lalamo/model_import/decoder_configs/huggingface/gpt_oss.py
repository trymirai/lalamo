from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DecoderLayerConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    MixtureOfExpertsConfig,
    RMSNormConfig,
    SoftmaxRouting,
    TiedEmbeddingConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
    YARNRoPEConfig,
)
from lalamo.modules.activations import SiLU

from .common import HuggingFaceConfig

__all__ = ["HFGPTOssConfig"]


@dataclass(frozen=True)
class YarnRopeScalingConfig:
    factor: float
    beta_fast: float
    beta_slow: float
    original_max_position_embeddings: int
    rope_type: Literal["yarn"]
    truncate: bool


@dataclass(frozen=True)
class HFGPTOssConfig(HuggingFaceConfig):
    # Core HF fields
    architectures: list[Literal["GptOssForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: Literal["gpt_oss"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    vocab_size: int

    # GPT-OSS specifics
    layer_types: list[Literal["sliding_attention", "full_attention"]] | None
    sliding_window: int | None
    swiglu_limit: float
    head_dim: int | None
    num_local_experts: int
    num_experts_per_tok: int | None = None
    experts_per_token: int | None = None  # some configs may use this alias
    rope_scaling: YarnRopeScalingConfig | None = None
    output_router_logits: bool | None = None
    router_aux_loss_coef: float | None = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        # Embedding
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

        if self.rope_scaling is not None and self.rope_scaling.rope_type == "yarn":
            rope_config = YARNRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
                original_context_length=self.rope_scaling.original_max_position_embeddings,
                beta_fast=self.rope_scaling.beta_fast,
                beta_slow=self.rope_scaling.beta_slow,
                truncate=self.rope_scaling.truncate,
            )
        else:
            rope_config = YARNRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
                scaling_factor=1.0,
                original_context_length=self.max_position_embeddings,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=True,
            )

        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
        )

        # Linear layers
        linear_config = FullPrecisionLinearConfig(precision=activation_precision)

        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            logit_soft_cap=None,
            has_sinks=True,
            has_qkv_biases=self.attention_bias,
            has_out_biases=self.attention_bias,
        )

        # Experts (MoE) scaffold
        # Router: linear with bias; Experts: DenseMLP with SiLU(alpha=1.702) and value/gate clipping
        experts_activation = SiLU(alpha=1.702)
        experts_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=experts_activation,
            has_up_biases=True,
            has_down_biases=True,
            up_clipping=(-self.swiglu_limit + 1.0, self.swiglu_limit + 1.0),
            gate_clipping=(None, self.swiglu_limit),
        )
        moe_config = MixtureOfExpertsConfig(
            mixture_size=self.num_local_experts,
            num_experts_per_token=(self.num_experts_per_tok or self.experts_per_token or 1),
            routing_function=SoftmaxRouting(),
            router_config=linear_config,
            router_has_biases=True,
            expert_config=experts_config,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=moe_config,
            post_mlp_norm_config=None,
        )

        # Per-layer sliding-window
        if self.layer_types is not None and len(self.layer_types) == self.num_hidden_layers:
            sliding_window_sizes = tuple(
                self.sliding_window if layer_type == "sliding_attention" else None for layer_type in self.layer_types
            )
        else:
            # Fallback: apply the same sliding window to all layers if provided
            sliding_window_sizes = (
                tuple([self.sliding_window] * self.num_hidden_layers) if self.sliding_window is not None else None
            )

        head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

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
            head_dim=head_dim,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=sliding_window_sizes,
            context_length=context_length or self.max_position_embeddings,
        )
