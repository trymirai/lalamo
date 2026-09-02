from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, MixtureOfExpertsConfig, SigmoidRouting
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import LlamaRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig

from .common import HuggingFaceLMConfig

__all__ = ["HFMobileMoEConfig"]


@dataclass(frozen=True)
class MobileMoERopeScalingConfig:
    rope_type: Literal["llama3"]
    factor: Literal[16]
    original_max_position_embeddings: Literal[8192]
    low_freq_factor: float
    high_freq_factor: float


@dataclass(frozen=True)
class MobileMoEQuantizationConfig:
    format: Literal["mobilemoe-int4-g32"]
    group_size: Literal[32]
    embedding_group_size: Literal[32]
    qmin: Literal[-8]
    qmax: Literal[7]
    symmetric: Literal[True]
    scale_dtype: Literal["float16"]
    linear_group_axis: Literal["in (last dim of [out, in])"]
    expert_group_axis: Literal["out (last dim of [E, in, out])"]
    packed: Literal[True]


@dataclass(frozen=True)
class HFMobileMoEConfig(HuggingFaceLMConfig):
    architectures: list[Literal["MobileMoEForCausalLM"]]
    eos_token_id: list[int]
    model_type: Literal["mobilemoe"]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    intermediate_size_mlp: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: Literal["silu"]
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: MobileMoERopeScalingConfig
    num_local_experts: int
    num_experts_per_tok: int
    interleave_moe_layer_step: int
    no_rope_layers: list[Literal[0, 1]]
    use_qk_norm: Literal[True]
    attn_temperature_tuning: Literal[False]
    tie_word_embeddings: Literal[True]
    torch_dtype: Literal["bfloat16"]
    attention_bias: Literal[False]
    routed_scaling_factor: float
    norm_topk_prob: Literal[True]
    quantization: MobileMoEQuantizationConfig

    def to_decoder_config(
        self,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        if self.interleave_moe_layer_step < 1:
            raise ValueError("MobileMoE interleave_moe_layer_step must be positive.")
        if self.intermediate_size_mlp % self.intermediate_size != 0:
            raise ValueError("MobileMoE shared expert width must be divisible by routed expert width.")

        rope_config = LlamaRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=self.head_dim,
            scaling_factor=self.rope_scaling.factor,
            original_context_length=self.rope_scaling.original_max_position_embeddings,
            low_frequency_factor=self.rope_scaling.low_freq_factor,
            high_frequency_factor=self.rope_scaling.high_freq_factor,
        )
        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        qk_norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
            has_scale=False,
        )
        linear_config = LinearConfig()
        expert_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )
        moe_config = MixtureOfExpertsConfig(
            expert_config=expert_config,
            router_config=linear_config,
            routing_function=SigmoidRouting(scale=self.routed_scaling_factor),
            num_routed_experts=self.num_local_experts,
            num_active_routed_experts=self.num_experts_per_tok,
            router_has_biases=False,
            num_shared_experts=self.intermediate_size_mlp // self.intermediate_size,
            expert_hidden_dim=self.intermediate_size,
        )
        attention_config = AttentionConfig(
            qkvg_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=qk_norm_config,
            key_norm_config=qk_norm_config,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkvg_biases=self.attention_bias,
            has_out_biases=self.attention_bias,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.head_dim,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
        )
        moe_layer_config = TransformerLayerConfig(
            pre_mixer_norm_config=rmsnorm_config,
            mixer_config=attention_config,
            post_mixer_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=moe_config,
            post_mlp_norm_config=None,
            rope_config=rope_config,
        )
        dense_layer_config = replace(
            moe_layer_config,
            mlp_config=expert_config,
            hidden_dim=self.intermediate_size_mlp,
        )
        transformer_config = TransformerConfig(
            layer_configs=tuple(
                replace(
                    moe_layer_config
                    if (layer_index + 1) % self.interleave_moe_layer_step == 0
                    else dense_layer_config,
                    mixer_config=attention_config
                    if use_rope
                    else replace(attention_config, query_norm_config=None, key_norm_config=None),
                    rope_config=rope_config if use_rope else None,
                )
                for layer_index, use_rope in zip(
                    range(self.num_hidden_layers),
                    self.no_rope_layers,
                    strict=True,
                )
            ),
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
        )
        return DecoderConfig(
            embedding_config=TiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
            ),
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
