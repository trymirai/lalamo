import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Self

import cattrs

from lalamo.modules.activations import SiLU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.speculators.dspark import (
    DSparkLayerConfig,
    DSparkSpeculatorConfig,
    MarkovHeadConfig,
    TapMixConfig,
)
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer_layer import SublayerConfig, TransformerLayerConfig

__all__ = ["HFDSparkConfig"]


@dataclass(frozen=True)
class HFDSparkConfig:
    """The `mtp_config` of a target that bundles a DSpark draft, with the target's vocabulary and logit soft cap."""

    _converter: ClassVar[cattrs.Converter] = cattrs.Converter(forbid_extra_keys=True)

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    torch_dtype: str
    q_bias: bool
    o_bias: bool
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: tuple[int, ...]
    sliding_window_size: int
    block_size: int
    markov_rank: int
    markov_head_type: Literal["rnn_hidden"]
    target_layer_ids: tuple[int, ...]
    vocab_size: int
    final_logit_softcapping: float | None

    @classmethod
    def from_target_json(cls, path: Path | str) -> Self:
        config = json.loads(Path(path).read_text())
        text_config = config["text_config"]
        return cls._converter.structure(
            {
                **config["mtp_config"],
                "vocab_size": text_config["vocab_size"],
                "final_logit_softcapping": text_config.get("final_logit_softcapping"),
            },
            cls,
        )

    @staticmethod
    def has_dspark(path: Path | str) -> bool:
        mtp_config = json.loads(Path(path).read_text()).get("mtp_config") or {}
        return mtp_config.get("markov_head_type") == "rnn_hidden"

    def to_dspark_config(self) -> DSparkSpeculatorConfig:
        if self.q_bias or self.o_bias:
            raise ValueError("DSpark drafts with projection biases are not supported.")
        linear_config = LinearConfig()
        norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        rope_config = UnscaledRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=self.head_dim,
        )
        tap_mix_config = TapMixConfig(num_taps=len(self.target_layer_ids), norm_config=norm_config)
        # Every layer, the full-attention one included, attends to the newest context rows and to the whole block.
        layer_config = TransformerLayerConfig(
            mixer=SublayerConfig(
                pre_norm_config=norm_config,
                conv_config=None,
                body_config=AttentionConfig(
                    qkvg_projection_config=linear_config,
                    out_projection_config=linear_config,
                    query_norm_config=norm_config,
                    key_norm_config=norm_config,
                    num_heads=self.num_attention_heads,
                    num_groups=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    is_causal=False,
                    scale=self.head_dim**-0.5,
                    sliding_window_size=None,
                    logit_soft_cap=None,
                    has_sinks=False,
                    has_qkvg_biases=False,
                    has_out_biases=False,
                ),
                post_norm_config=None,
            ),
            mlp=SublayerConfig(
                pre_norm_config=norm_config,
                conv_config=None,
                body_config=DenseMLPConfig(
                    linear_config=linear_config,
                    activation=SiLU(),
                    has_up_biases=False,
                    has_down_biases=False,
                    gate_clipping=None,
                    up_clipping=None,
                ),
                post_norm_config=None,
            ),
            rope_config=rope_config,
        )
        dspark_layer_config = DSparkLayerConfig(
            transformer_layer_config=layer_config,
            context_mix_config=tap_mix_config,
            context_kv_projection_config=linear_config,
            context_key_norm_config=norm_config,
        )
        return DSparkSpeculatorConfig(
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            block_size=self.block_size,
            target_feature_indices=self.target_layer_ids,
            context_window=self.sliding_window_size - 1,
            rope_config=rope_config,
            normalizes_anchor_embedding=True,
            layer_configs=(dspark_layer_config,) * self.num_hidden_layers,
            output_norm_config=norm_config,
            markov_head_config=MarkovHeadConfig(
                rank=self.markov_rank,
                vocab_size=self.vocab_size,
                context_mix_config=tap_mix_config,
                recurrence_config=linear_config,
                output_config=linear_config,
                logit_soft_cap=self.final_logit_softcapping,
            ),
        )
