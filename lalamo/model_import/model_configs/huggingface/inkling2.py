import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import equinox as eqx
from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders.inkling2 import load_inkling2_decoder
from lalamo.models import LanguageModel
from lalamo.modules.activations import SoftCappedSiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, MixtureOfExpertsConfig, RoutingBias, SigmoidRouting
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.residual import BlockAttentionResidualConfig, MessageGateConfig
from lalamo.modules.token_mixers.attention import (
    AttentionConfig,
    L2ClipConfig,
    LogScalingConfig,
    RelativePositionBiasConfig,
)
from lalamo.modules.token_mixers.convolutions import CausalConvConfig
from lalamo.modules.transformer import LoopConfig, TransformerConfig
from lalamo.modules.transformer_layer import SublayerConfig, TransformerLayerConfig
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig

__all__ = ["HFInkling2LoopConfig", "Inkling2LoopParameters"]


@dataclass(frozen=True)
class Inkling2LoopParameters:
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    is_varloop: Literal[False]


@dataclass(frozen=True)
class HFInkling2LoopConfig(HuggingFaceLMConfig):
    """Inkling 2 Mini as converted to MLX by mlx-lm, with the text model's settings under `text_config`."""

    model_type: Literal["inkling2_loop"]
    eos_token_id: int

    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    rms_norm_eps: float

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    local_layer_ids: list[int]
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int
    log_scaling_alpha: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    sconv_kernel_size: int
    q_bias: Literal[False]
    o_bias: Literal[False]
    use_value_norm_no_gamma: Literal[True]

    n_layers_per_attn_res_block: int
    attn_res_normalize_values: Literal[True]

    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: Literal[0]
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    use_gate_bias: Literal[True]

    final_logit_softcapping: float
    model_max_length: int
    loop_config: Inkling2LoopParameters

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        with open(json_path) as json_file:
            config = json.load(json_file)
        text_config = config.pop("text_config", {})
        return cls._converter.structure({**text_config, **config}, cls)

    def _attention_config(self, *, is_local: bool) -> AttentionConfig:
        linear_config = LinearConfig()
        kernel_size = self.sconv_kernel_size
        if is_local:
            num_heads, num_groups, head_dim = (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            )
            conv_config = CausalConvConfig(
                kernel_size=kernel_size,
                num_dynamic_groups=None,
                has_identity=True,
                output_scale=1 / math.sqrt(kernel_size + 1),
            )
        else:
            num_heads, num_groups, head_dim = self.num_attention_heads, self.num_key_value_heads, self.head_dim
            conv_config = CausalConvConfig(
                kernel_size=kernel_size,
                num_dynamic_groups=num_groups,
                has_identity=True,
                output_scale=1 / math.sqrt(2 * kernel_size + 1),
            )
        return AttentionConfig(
            qkvg_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            is_causal=True,
            scale=None,
            sliding_window_size=self.sliding_window_size if is_local else None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkvg_biases=False,
            has_out_biases=False,
            has_gate=True,
            normalize_values=True,
            separate_inputs=True,
            query_clip_config=L2ClipConfig(threshold=self.qk_l2_clip_threshold, has_scale=True),
            key_clip_config=L2ClipConfig(threshold=self.qk_l2_clip_threshold, has_scale=False),
            log_scaling_config=(
                None if is_local else LogScalingConfig(n_floor=self.log_scaling_n_floor, alpha=self.log_scaling_alpha)
            ),
            relative_position_bias_config=RelativePositionBiasConfig(
                dim=self.d_rel,
                extent=self.sliding_window_size if is_local else self.rel_extent,
                clip_config=L2ClipConfig(threshold=self.rel_l2_clip_threshold, has_scale=True),
            ),
            has_token_sinks=True,
            key_conv_config=conv_config,
            value_conv_config=conv_config,
        )

    def to_decoder_config(
        self,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        linear_config = LinearConfig()
        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )
        output_conv_config = CausalConvConfig(
            kernel_size=self.sconv_kernel_size,
            num_dynamic_groups=None,
            has_identity=True,
            output_scale=1 / math.sqrt(self.sconv_kernel_size + 1),
        )
        mlp_config = MixtureOfExpertsConfig(
            expert_config=DenseMLPConfig(
                linear_config=linear_config,
                activation=SoftCappedSiLU(soft_cap=self.mlp_gate_softcap),
                has_up_biases=False,
                has_down_biases=False,
                gate_clipping=None,
                up_clipping=None,
                up_soft_cap=self.mlp_up_softcap,
            ),
            router_config=linear_config,
            routing_function=SigmoidRouting(bias_mode=RoutingBias.GATING, routed_scaling_factor=self.route_scale),
            num_routed_experts=self.n_routed_experts,
            num_active_routed_experts=self.num_experts_per_tok,
            router_has_biases=False,
            num_shared_experts=self.n_shared_experts,
            expert_hidden_dim=self.intermediate_size,
            has_float32_router=True,
        )
        layer_configs = tuple(
            TransformerLayerConfig(
                mixer=SublayerConfig(
                    pre_norm_config=rmsnorm_config,
                    conv_config=None,
                    body_config=self._attention_config(is_local=layer_index in self.local_layer_ids),
                    post_norm_config=None,
                    num_inputs=3,
                    separate_input_norms=layer_index not in self.local_layer_ids,
                    output_conv_config=output_conv_config,
                ),
                mlp=SublayerConfig(
                    pre_norm_config=rmsnorm_config,
                    conv_config=None,
                    body_config=mlp_config,
                    post_norm_config=None,
                    output_conv_config=output_conv_config,
                ),
            )
            for layer_index in range(self.num_hidden_layers)
        )
        return DecoderConfig(
            embedding_config=UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=self.final_logit_softcapping),
            transformer_config=TransformerConfig(
                residual_config=BlockAttentionResidualConfig(
                    block_size=self.n_layers_per_attn_res_block,
                    epsilon=self.rms_norm_eps,
                    message_gate_config=MessageGateConfig(
                        rank=self.loop_config.gate_rank, linear_config=linear_config
                    ),
                ),
                layer_configs=layer_configs,
                output_norm_config=rmsnorm_config,
                model_dim=self.hidden_size,
                hidden_dim=self.intermediate_size,
                loop_config=LoopConfig(
                    start_layer=self.loop_config.core_start,
                    end_layer=self.loop_config.core_end,
                    num_passes=self.loop_config.num_passes,
                ),
            ),
            vocab_size=self.unpadded_vocab_size,
            embedding_norm_config=rmsnorm_config,
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        assert isinstance(model, LanguageModel)
        decoder = load_inkling2_decoder(model.decoder, weights_dict, implementation=implementation)
        return eqx.tree_at(lambda m: (m.decoder,), model, (decoder,))
