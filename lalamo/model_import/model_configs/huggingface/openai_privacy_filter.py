from dataclasses import dataclass
from typing import Any, Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    ClassifierConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    NormalizationConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UpcastMode,
)
from lalamo.modules.activations import SiLU
from lalamo.modules.classifier import PoolingType, PredictionHeadConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.mlp import MixtureOfExpertsConfig, SoftmaxRouting
from lalamo.modules.rope import YARNRoPEConfig

from .common import HuggingFaceClassifierConfig

__all__ = ["HFOpenAIPrivacyFilterConfig"]


@dataclass(frozen=True)
class HFOpenAIPrivacyFilterConfig(HuggingFaceClassifierConfig):
    """Config for `openai/privacy-filter` — a bidirectional token-classification
    model with GQA + attention sinks, sliding-window attention, YaRN rope, and
    an MoE feed-forward (128 experts, top-4 routing). Outputs 33 BIOES-style
    PII labels per token.
    """

    architectures: list[Literal["OpenAIPrivacyFilterForTokenClassification"]]
    model_type: Literal["openai_privacy_filter"]

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    vocab_size: int
    sliding_window: int

    attention_bias: bool
    attention_dropout: float
    classifier_dropout: float
    tie_word_embeddings: bool

    # MoE
    num_local_experts: int
    num_experts_per_tok: int

    # RoPE (YaRN)
    rope_parameters: dict[str, Any]

    # Labels
    id2label: dict[int, str]
    label2id: dict[str, int]

    # Housekeeping / tolerated extras
    transformers_version: str = ""
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    initial_context_length: int | None = None
    default_n_ctx: int | None = None
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = True
    hidden_act: Literal["silu"] = "silu"

    def __post_init__(self) -> None:
        if len(self.id2label) != len(self.label2id):
            raise ValueError("id2label and label2id must agree on cardinality.")

    @property
    def default_precision(self) -> DTypeLike:
        import jax.numpy as jnp

        return jnp.dtype(self.dtype)

    def to_classifier_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> ClassifierConfig:
        linear_config = FullPrecisionLinearConfig(precision=activation_precision)

        embedding_config = TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=activation_precision,
        )

        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        yarn_rope_config = YARNRoPEConfig(
            precision=activation_precision,
            base=float(self.rope_parameters["rope_theta"]),
            max_sequence_length=context_length or self.max_position_embeddings,
            head_dim=self.head_dim,
            scaling_factor=float(self.rope_parameters["factor"]),
            original_context_length=int(self.rope_parameters["original_max_position_embeddings"]),
            beta_fast=float(self.rope_parameters["beta_fast"]),
            beta_slow=float(self.rope_parameters["beta_slow"]),
            truncate=bool(self.rope_parameters.get("truncate", False)),
        )

        # Each expert is GPT-OSS-style: `(up+1) * gate * sigmoid(gate * 1.702)`
        # with `gate` clipped to (-inf, 7] and `up` clipped to [-7, 7] BEFORE the +1.
        # Lalamo bakes the +1 into up_bias at load time (see huggingface.py
        # batched MoE loader) and shifts the up clipping to the post-+1 range
        # [-6, 8] so `up_proj_clipped = clamp(Wx + (b+1), -6, 8)` equals the HF
        # `clamp(Wx + b, -7, 7) + 1`. `swiglu_limit` is hard-coded to 7.0 here
        # (matches the com.microsoft.MoE attribute on the ONNX export).
        _swiglu_limit = 7.0
        expert_mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(alpha=1.702),
            has_up_biases=True,
            has_down_biases=True,
            up_clipping=(-_swiglu_limit + 1.0, _swiglu_limit + 1.0),
            gate_clipping=(None, _swiglu_limit),
        )
        moe_config = MixtureOfExpertsConfig(
            num_routed_experts=self.num_local_experts,
            num_active_routed_experts=self.num_experts_per_tok,
            routing_function=SoftmaxRouting(),
            router_config=linear_config,
            router_has_biases=True,
            expert_config=expert_mlp_config,
            gate_config=None,
            num_shared_experts=0,
            expert_hidden_dim=self.intermediate_size,
        )

        layer_configs = []
        for _ in range(self.num_hidden_layers):
            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=None,
                has_sinks=True,
                has_qkv_biases=self.attention_bias,
                has_out_biases=self.attention_bias,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=False,
                scale=None,
                sliding_window_size=self.sliding_window,
                gate_projection_config=None,
            )
            layer_configs.append(
                TransformerLayerConfig(
                    pre_mixer_norm_config=rmsnorm_config,
                    mixer_config=attention_config,
                    post_mixer_norm_config=None,
                    pre_mlp_norm_config=rmsnorm_config,
                    mlp_config=moe_config,
                    post_mlp_norm_config=None,
                    rope_config=yarn_rope_config,
                ),
            )

        transformer_config = TransformerConfig(
            layer_configs=tuple(layer_configs),
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=context_length or self.max_position_embeddings,
        )

        prediction_head_config = PredictionHeadConfig(
            readout_config=linear_config,
            readout_has_biases=True,
        )

        output_labels = tuple(self.id2label[idx] for idx in range(len(self.id2label)))

        return ClassifierConfig(
            embedding_config=embedding_config,
            embedding_norm_config=None,
            transformer_config=transformer_config,
            prediction_head_config=prediction_head_config,
            readout_config=linear_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            context_length=context_length or self.max_position_embeddings,
            num_labels=len(self.id2label),
            classifier_pooling=PoolingType.NONE,
            output_labels=output_labels,
        )
