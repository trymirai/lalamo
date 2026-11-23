import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import equinox as eqx
from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders.inkling import load_inkling2_loop_decoder
from lalamo.models import LanguageModel
from lalamo.modules.activations import SoftCappedSiLU
from lalamo.modules.attention_residual import (
    AttnResQKVReadConfig,
    AttnResReadConfig,
    BlockAttnResConfig,
    MessageGateConfig,
    QKVReadMode,
)
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, MixtureOfExpertsConfig, RoutingBias, SigmoidRouting
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.attention import (
    AttentionConfig,
    ConvolvedAttentionConfig,
    KeyValueLayout,
    LogScalingConfig,
    RelativeBiasConfig,
    SinkSource,
    aligned_rows,
)
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import OutputConvConfig, SublayerConfig, TransformerLayerConfig
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig

__all__ = ["HFInkling2LoopConfig"]

ROW_ALIGNMENT = 128


@dataclass(frozen=True)
class Inkling2LoopPassConfig:
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    is_varloop: Literal[False]


@dataclass(frozen=True)
class HFInkling2LoopConfig(HuggingFaceLMConfig):
    model_type: Literal["inkling2_loop"]
    eos_token_id: int
    hidden_size: int
    num_hidden_layers: int
    unpadded_vocab_size: int
    rms_norm_eps: float
    final_logit_softcapping: float

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
    loop_config: Inkling2LoopPassConfig

    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: Literal[0]
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    use_gate_bias: Literal[True]

    def __post_init__(self) -> None:
        if self.rms_norm_eps != 1e-6:
            raise ValueError(f"The value norm of lalamo attention uses epsilon 1e-6, got {self.rms_norm_eps}.")
        block_size = self.n_layers_per_attn_res_block
        if self.loop_config.core_start % block_size != 0 or self.loop_config.core_end % block_size != 0:
            raise ValueError(
                f"Loop bounds {self.loop_config.core_start}..{self.loop_config.core_end} must lie on"
                f" attention residual block boundaries of size {block_size}.",
            )

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        with open(json_path) as json_file:
            config = json.load(json_file)
        text_config = config.pop("text_config")
        return cls._converter.structure({**text_config, **config}, cls)

    @property
    def layer_order(self) -> tuple[int, ...]:
        loop_layers = tuple(range(self.loop_config.core_start, self.loop_config.core_end))
        return (
            *range(self.loop_config.core_start),
            *loop_layers * self.loop_config.num_passes,
            *range(self.loop_config.core_end, self.num_hidden_layers),
        )

    @property
    def norm_config(self) -> NormalizationConfig:
        return NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
            has_scale=False,
        )

    def layer_config(self, layer_index: int) -> TransformerLayerConfig:
        linear_config = LinearConfig()
        conv_config = SeparableCausalConvConfig(has_biases=False)
        is_sliding = layer_index in self.local_layer_ids
        is_looped = self.loop_config.core_start <= layer_index < self.loop_config.core_end
        message_gate_config = (
            MessageGateConfig(linear_config=linear_config, rank=self.loop_config.gate_rank) if is_looped else None
        )
        if layer_index == 0:
            read_mode = QKVReadMode.PARTIAL
        elif is_sliding:
            read_mode = QKVReadMode.SHARED
        else:
            read_mode = QKVReadMode.SEPARATE
        head_dim = self.swa_head_dim if is_sliding else self.head_dim
        attention_config = ConvolvedAttentionConfig(
            attention_config=AttentionConfig(
                qkvg_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                num_heads=self.swa_num_attention_heads if is_sliding else self.num_attention_heads,
                num_groups=self.swa_num_key_value_heads if is_sliding else self.num_key_value_heads,
                head_dim=head_dim,
                is_causal=True,
                scale=1.0,
                sliding_window_size=self.sliding_window_size if is_sliding else None,
                logit_soft_cap=None,
                has_sinks=True,
                has_qkvg_biases=self.q_bias,
                has_out_biases=self.o_bias,
                has_gate=True,
                normalize_values=self.use_value_norm_no_gamma,
                sink_source=SinkSource.PROJECTED,
                relative_bias_config=RelativeBiasConfig(
                    num_channels=self.d_rel,
                    extent=self.sliding_window_size if is_sliding else self.rel_extent,
                    clip_threshold=self.rel_l2_clip_threshold,
                ),
                query_clip_threshold=self.qk_l2_clip_threshold * head_dim**-0.5,
                key_clip_threshold=self.qk_l2_clip_threshold,
                log_scaling_config=(
                    None
                    if is_sliding
                    else LogScalingConfig(alpha=self.log_scaling_alpha, position_floor=self.log_scaling_n_floor)
                ),
            ),
            conv_config=conv_config,
            key_value_layout=KeyValueLayout.STACKED_WITH_QUERY
            if is_sliding
            else KeyValueLayout.GROUPED_WITH_CONV_TAPS,
            conv_kernel_size=self.sconv_kernel_size,
            row_alignment=ROW_ALIGNMENT,
        )
        output_conv_config = OutputConvConfig(conv_config=conv_config, kernel_size=self.sconv_kernel_size)
        moe_config = MixtureOfExpertsConfig(
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
            routing_function=SigmoidRouting(bias_mode=RoutingBias.GATING, weight_scale=self.route_scale),
            num_routed_experts=self.n_routed_experts,
            num_active_routed_experts=self.num_experts_per_tok,
            router_has_biases=False,
            num_shared_experts=self.n_shared_experts,
            expert_hidden_dim=self.intermediate_size,
        )
        return TransformerLayerConfig(
            mixer=SublayerConfig(
                pre_norm_config=self.norm_config,
                conv_config=None,
                body_config=attention_config,
                post_norm_config=None,
                connection_config=AttnResQKVReadConfig(
                    read_mode=read_mode,
                    starts_block=layer_index % self.n_layers_per_attn_res_block == 0,
                    message_gate_config=message_gate_config,
                ),
                output_conv_config=output_conv_config,
            ),
            mlp=SublayerConfig(
                pre_norm_config=self.norm_config,
                conv_config=None,
                body_config=moe_config,
                post_norm_config=None,
                connection_config=AttnResReadConfig(message_gate_config=message_gate_config),
                output_conv_config=output_conv_config,
            ),
        )

    def to_decoder_config(
        self,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        return DecoderConfig(
            embedding_config=UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=self.final_logit_softcapping),
            transformer_config=TransformerConfig(
                residual_config=BlockAttnResConfig(epsilon=self.rms_norm_eps),
                layer_configs=tuple(self.layer_config(layer_index) for layer_index in range(self.num_hidden_layers)),
                output_norm_config=self.norm_config,
                model_dim=self.hidden_size,
                hidden_dim=self.intermediate_size,
                layer_order=self.layer_order,
            ),
            vocab_size=aligned_rows(self.unpadded_vocab_size, ROW_ALIGNMENT),
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
    ) -> Model:
        assert isinstance(model, LanguageModel)
        decoder = load_inkling2_loop_decoder(
            model.decoder,
            weights_dict,
            loop_start=self.loop_config.core_start,
            unpadded_vocab_size=self.unpadded_vocab_size,
        )
        return eqx.tree_at(lambda m: (m.decoder,), model, (decoder,))
