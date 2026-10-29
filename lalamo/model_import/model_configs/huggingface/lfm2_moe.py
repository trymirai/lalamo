from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.loaders.huggingface import load_mlp
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, MixtureOfExpertsConfig, RoutingBias, SigmoidRouting
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.short_conv import ShortConvConfig

from .common import HuggingFaceLMConfig, residual_decoder
from .lfm2 import RopeParameters


@dataclass(frozen=True)
class HFLFM2MoeConfig(HuggingFaceLMConfig):
    architectures: list[Literal["Lfm2MoeForCausalLM"]]
    bos_token_id: int
    conv_L_cache: int  # noqa: N815
    conv_bias: Literal[False]
    dtype: Literal["bfloat16"]
    eos_token_id: int
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    layer_types: list[Literal["conv", "full_attention"]]
    max_position_embeddings: int
    model_type: Literal["lfm2_moe"]
    moe_intermediate_size: int
    norm_eps: float
    norm_topk_prob: Literal[True]
    num_attention_heads: int
    num_dense_layers: int
    num_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    rope_parameters: RopeParameters
    routed_scaling_factor: float
    tie_word_embeddings: Literal[True]
    transformers_version: str
    use_cache: bool
    use_expert_bias: Literal[True]
    vocab_size: int

    def build_decoder(self, at: Binder) -> DecoderConfig:
        assert self.rope_parameters.rope_type == "default"
        assert self.routed_scaling_factor == 1.0
        assert len(self.layer_types) == self.num_hidden_layers
        head_dim = self.hidden_size // self.num_attention_heads
        linear_config = LinearConfig()
        norm_config = NormalizationConfig(
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        rope_config = UnscaledRoPEConfig(
            base=self.rope_parameters.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=head_dim,
        )
        attention_config = AttentionConfig(
            qkvg_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=norm_config,
            key_norm_config=norm_config,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=head_dim,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkvg_biases=False,
            has_out_biases=False,
            rope_config=rope_config,
        )
        conv_config = ShortConvConfig(
            in_projection_config=linear_config,
            conv_config=SeparableCausalConvConfig(has_biases=self.conv_bias),
            out_projection_config=linear_config,
            kernel_size=self.conv_L_cache,
        )
        dense_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
            hidden_dim=self.intermediate_size,
        )
        moe_config = MixtureOfExpertsConfig(
            expert_config=replace(dense_config, hidden_dim=self.moe_intermediate_size),
            router_config=linear_config,
            # LFM's expert bias affects selection only, not the mixture weights.
            routing_function=SigmoidRouting(bias_mode=RoutingBias.GATING),
            num_routed_experts=self.num_experts,
            num_active_routed_experts=self.num_experts_per_tok,
            router_has_biases=False,
            num_shared_experts=0,
        )
        mixer_configs = {"conv": conv_config, "full_attention": attention_config}
        mixer_keys = {"conv": "conv", "full_attention": "self_attn"}
        load_feed_forward = partial(load_mlp, up_proj_key="w3", gate_proj_key="w1", down_proj_key="w2")
        return residual_decoder(
            at,
            embedding=at("model.embed_tokens", TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None)),
            layers=[
                (
                    at(f"model.layers.{index}.{mixer_keys[layer_type]}", mixer_configs[layer_type]),
                    at(
                        f"model.layers.{index}.feed_forward",
                        dense_config if index < self.num_dense_layers else moe_config,
                        load=load_feed_forward,
                    ),
                )
                for index, layer_type in enumerate(self.layer_types)
            ],
            norm=norm_config,
            model_dim=self.hidden_size,
            vocab_size=self.vocab_size,
            mixer_norm="operator_norm",
            mlp_norm="ffn_norm",
            output_norm="embedding_norm",
        )
