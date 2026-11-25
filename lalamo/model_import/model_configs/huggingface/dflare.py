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
from lalamo.modules.speculators.dflare import DFlareDraftConfig
from lalamo.modules.speculators.dspark import DSparkConfig, MarkovHeadType
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer_layer import SublayerConfig, TransformerLayerConfig

from .dflash import HFDFlashConfig
from .inkling2_loop import HFInkling2LoopConfig

__all__ = [
    "HFDFlareConfig",
    "HFDFlareDraftConfig",
    "load_hf_speculator_config",
]


@dataclass(frozen=True)
class HFDFlareDraftConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    torch_dtype: Literal["bfloat16"]
    q_bias: Literal[False]
    o_bias: Literal[False]
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: list[int]
    sliding_window_size: int
    block_size: int
    markov_rank: int
    markov_head_type: MarkovHeadType
    target_layer_ids: list[int]

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> Self:
        return cls._converter.structure(config, cls)


@dataclass(frozen=True)
class HFDFlareConfig:
    target_config: HFInkling2LoopConfig
    draft_config: HFDFlareDraftConfig

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        with open(json_path) as json_file:
            draft_config = json.load(json_file)["mtp_config"]
        return cls(
            target_config=HFInkling2LoopConfig.from_json(json_path),
            draft_config=HFDFlareDraftConfig.from_dict(draft_config),
        )

    @property
    def norm_config(self) -> NormalizationConfig:
        return NormalizationConfig(
            epsilon=self.draft_config.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )

    @property
    def vocab_size(self) -> int:
        return self.target_config.to_decoder_config({}).vocab_size

    def to_dflare_draft_config(self) -> DFlareDraftConfig:
        draft = self.draft_config
        target = self.target_config
        num_steps = len(target.layer_order)
        if draft.hidden_size != target.hidden_size:
            raise ValueError(f"The DFlare draft width {draft.hidden_size} differs from {target.hidden_size}.")
        first_layer_id, *_, last_layer_id = draft.target_layer_ids
        if (
            draft.target_layer_ids != sorted(set(draft.target_layer_ids))
            or first_layer_id < 1
            or last_layer_id != num_steps
        ):
            raise ValueError(
                f"DFlare target_layer_ids {draft.target_layer_ids} must be increasing layer visits ending at"
                f" the final output {num_steps}.",
            )
        linear_config = LinearConfig()
        norm_config = self.norm_config
        rope_config = UnscaledRoPEConfig(
            base=draft.rope_theta,
            max_sequence_length=draft.max_position_embeddings,
            head_dim=draft.head_dim,
        )
        layer_configs = tuple(
            TransformerLayerConfig(
                mixer=SublayerConfig.pre_norm(
                    norm_config,
                    AttentionConfig(
                        qkvg_projection_config=linear_config,
                        out_projection_config=linear_config,
                        query_norm_config=norm_config,
                        key_norm_config=norm_config,
                        num_heads=draft.num_attention_heads,
                        num_groups=draft.num_key_value_heads,
                        head_dim=draft.head_dim,
                        is_causal=False,
                        scale=draft.head_dim**-0.5,
                        sliding_window_size=(
                            2 * draft.sliding_window_size if layer_index in draft.local_layer_ids else None
                        ),
                        logit_soft_cap=None,
                        has_sinks=False,
                        has_qkvg_biases=draft.q_bias,
                        has_out_biases=draft.o_bias,
                    ),
                ),
                mlp=SublayerConfig.pre_norm(
                    norm_config,
                    DenseMLPConfig(
                        linear_config=linear_config,
                        activation=SiLU(),
                        has_up_biases=False,
                        has_down_biases=False,
                        gate_clipping=None,
                        up_clipping=None,
                    ),
                ),
                rope_config=rope_config,
            )
            for layer_index in range(draft.num_hidden_layers)
        )
        return DFlareDraftConfig(
            model_dim=draft.hidden_size,
            hidden_dim=draft.intermediate_size,
            block_size=draft.block_size,
            target_layer_ids=tuple(layer_id - 1 for layer_id in draft.target_layer_ids),
            num_target_layers=num_steps,
            vocab_size=self.vocab_size,
            context_projection_config=linear_config,
            context_norm_config=norm_config,
            rope_config=rope_config,
            layer_configs=layer_configs,
            output_norm_config=norm_config,
        )

    def to_dspark_config(self) -> DSparkConfig:
        return DSparkConfig(
            model_dim=self.draft_config.hidden_size,
            vocab_size=self.vocab_size,
            num_context_taps=len(self.draft_config.target_layer_ids),
            markov_rank=self.draft_config.markov_rank,
            markov_head_type=self.draft_config.markov_head_type,
            projection_config=LinearConfig(),
            norm_config=self.norm_config,
            logit_soft_cap=self.target_config.final_logit_softcapping,
        )


def load_hf_speculator_config(json_path: Path | str) -> HFDFlashConfig | HFDFlareConfig:
    with open(json_path) as json_file:
        config = json.load(json_file)
    if "mtp_config" in config:
        return HFDFlareConfig.from_json(json_path)
    return HFDFlashConfig.from_dict(config)
