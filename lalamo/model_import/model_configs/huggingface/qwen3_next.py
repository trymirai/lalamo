from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig, UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, MixtureOfExpertsConfig, SoftmaxRouting
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNetConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig

from .common import HuggingFaceLMConfig, QuantizationConfigType

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

    quantization: QuantizationConfigType | None = None
    quantization_config: QuantizationConfigType | None = None

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
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

        if self.rope_scaling is not None:
            raise NotImplementedError("rope_scaling is not supported yet")
        rope_config = UnscaledRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=max_sequence_length,
            head_dim=int(self.head_dim * self.partial_rotary_factor),
        )

        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        # Gated DeltaNet RMSNorm in Qwen3-Next uses direct weights (no +1 offset).
        gated_rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        linear_config = LinearConfig()

        moe_config: MixtureOfExpertsConfig | None = None
        if self.num_experts > 0:
            assert self.shared_expert_intermediate_size == self.moe_intermediate_size, (
                f"shared_expert_intermediate_size ({self.shared_expert_intermediate_size}) "
                f"must equal moe_intermediate_size ({self.moe_intermediate_size})"
            )
            experts_config = DenseMLPConfig(
                linear_config=linear_config,
                activation=SiLU(),
                has_up_biases=False,
                has_down_biases=False,
                up_clipping=None,
                gate_clipping=None,
            )
            moe_config = MixtureOfExpertsConfig(
                num_routed_experts=self.num_experts,
                num_active_routed_experts=self.num_experts_per_tok,
                routing_function=SoftmaxRouting(),
                router_config=linear_config,
                router_has_biases=False,
                expert_config=experts_config,
                gate_config=linear_config,
                num_shared_experts=1,
                expert_hidden_dim=self.moe_intermediate_size,
            )

        layer_configs = []
        for layer_idx in range(self.num_hidden_layers):
            is_linear_attention = (layer_idx + 1) % self.full_attention_interval != 0
            if is_linear_attention:
                mixer_config = DeltaNetConfig(
                    in_proj_config=linear_config,
                    conv_config=SeparableCausalConvConfig(
                        has_biases=False,
                    ),
                    out_proj_config=linear_config,
                    norm_config=gated_rmsnorm_config,
                    num_heads=self.linear_num_value_heads,
                    num_groups=self.linear_num_key_heads,
                    head_dim=self.linear_key_head_dim,
                    value_head_dim=self.linear_value_head_dim,
                    kernel_size=self.linear_conv_kernel_dim,
                )
            else:
                mixer_config = AttentionConfig(
                    qkv_projection_config=linear_config,
                    out_projection_config=linear_config,
                    query_norm_config=rmsnorm_config,
                    key_norm_config=rmsnorm_config,
                    rope_config=rope_config,
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
                    gate_projection_config=linear_config,
                )

            if (
                moe_config is not None
                and layer_idx not in self.mlp_only_layers
                and (layer_idx + 1) % self.decoder_sparse_step == 0
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
            layer_configs=tuple(layer_configs),
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
