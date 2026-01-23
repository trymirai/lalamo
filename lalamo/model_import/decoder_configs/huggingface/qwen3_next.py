from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DeltaNetAttentionConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    MLXQuantizedTiedEmbeddingConfig,
    MLXQuantizedUntiedEmbeddingConfig,
    NormalizationConfig,
    SparseMoEConfig,
    TopKRouting,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
)
from lalamo.modules.activations import SiLU
from lalamo.modules.linear import MLXQuantizedLinearConfig
from lalamo.modules.token_mixers import SeparableCausalConvConfig
from lalamo.quantization import QuantizationMode

from .common import HuggingFaceLMConfig, MLXQuantizationConfig

__all__ = ["HFQwen3NextConfig"]


@dataclass(frozen=True)
class HFQwen3NextConfig(HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[Literal["Qwen3NextForCausalLM"]]
    attention_dropout: float
    bos_token_id: int | list[int]
    decoder_sparse_step: int
    eos_token_id: int | list[int]
    full_attention_interval: int
    head_dim: int
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    max_position_embeddings: int
    mlp_only_layers: list[int]
    model_type: Literal["qwen3_next"]
    moe_intermediate_size: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    output_router_logits: bool
    partial_rotary_factor: float
    rms_norm_eps: float
    rope_scaling: object | None
    rope_theta: float
    router_aux_loss_coef: float
    shared_expert_intermediate_size: int
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int

    attention_bias: bool = False

    quantization: object | None = None
    quantization_config: object | None = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        quantization = self.quantization or self.quantization_config
        if isinstance(quantization, dict):
            if "group_size" in quantization and "bits" in quantization:
                quantization = MLXQuantizationConfig(
                    group_size=int(quantization["group_size"]),
                    bits=int(quantization["bits"]),
                )
            else:
                quantization = None
        if isinstance(quantization, MLXQuantizationConfig):
            if self.tie_word_embeddings:
                embedding_config = MLXQuantizedTiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    group_size=quantization.group_size,
                    embedding_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                    activation_quantization_mode=None,
                    activation_precision=activation_precision,
                )
            else:
                embedding_config = MLXQuantizedUntiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    group_size=quantization.group_size,
                    embedding_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                    activation_quantization_mode=None,
                    activation_precision=activation_precision,
                )
        else:  # noqa: PLR5501
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

        if self.rope_scaling is not None:
            raise NotImplementedError("rope_scaling is not supported yet")
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=context_length or self.max_position_embeddings,
        )

        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        if quantization is None:
            linear_config = FullPrecisionLinearConfig(
                precision=activation_precision,
            )
        elif isinstance(quantization, MLXQuantizationConfig):
            linear_config = MLXQuantizedLinearConfig(
                group_size=quantization.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )
        else:
            linear_config = GroupQuantizedLinearConfig(
                group_size=quantization.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )

        experts_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )
        shared_expert_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )
        moe_config = SparseMoEConfig(
            mixture_size=self.num_experts,
            num_experts_per_token=self.num_experts_per_tok,
            routing_function=TopKRouting(norm_topk_prob=self.norm_topk_prob),
            router_config=linear_config,
            router_has_biases=False,
            expert_config=experts_config,
            shared_expert_config=shared_expert_config,
            shared_expert_gate_config=linear_config,
            expert_hidden_dim=self.moe_intermediate_size,
            shared_expert_hidden_dim=self.shared_expert_intermediate_size,
        )

        layer_types = [
            ("full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention")
            for i in range(self.num_hidden_layers)
        ]

        layer_configs = []
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "linear_attention":
                mixer_config = DeltaNetAttentionConfig(
                    in_proj_qkvz_config=linear_config,
                    in_proj_ba_config=linear_config,
                    conv_config=SeparableCausalConvConfig(
                        precision=activation_precision,
                        has_biases=False,
                    ),
                    out_proj_config=linear_config,
                    norm_config=rmsnorm_config,
                    num_k_heads=self.linear_num_key_heads,
                    num_v_heads=self.linear_num_value_heads,
                    head_k_dim=self.linear_key_head_dim,
                    head_v_dim=self.linear_value_head_dim,
                    kernel_size=self.linear_conv_kernel_dim,
                )
            else:
                mixer_config = AttentionConfig(
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
                    uses_separate_qkv=True,
                    q_proj_config=linear_config,
                    k_proj_config=linear_config,
                    v_proj_config=linear_config,
                    o_proj_config=linear_config,
                    q_proj_has_gate=True,
                    rope_dim_override=int(self.head_dim * self.partial_rotary_factor),
                )

            if (layer_idx not in self.mlp_only_layers) and (
                self.num_experts > 0 and (layer_idx + 1) % self.decoder_sparse_step == 0
            ):
                mlp_config = moe_config
            else:
                mlp_config = DenseMLPConfig(
                    linear_config=linear_config,
                    activation=SiLU(),
                    has_up_biases=False,
                    has_down_biases=False,
                    up_clipping=None,
                    gate_clipping=None,
                )

            transformer_layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=rmsnorm_config,
                mixer_config=mixer_config,
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
