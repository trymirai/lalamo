import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Self

import cattrs

from lalamo.modules.activations import SiLU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import RoPEConfig, UnscaledRoPEConfig, YARNRoPEConfig
from lalamo.modules.speculators.dflash import DFlashDraftConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig

__all__ = [
    "DFlashYarnRopeScalingConfig",
    "HFDFlashConfig",
    "HFDFlashInnerConfig",
]


@dataclass(frozen=True)
class DFlashYarnRopeScalingConfig:
    factor: float
    beta_fast: float
    beta_slow: float
    original_max_position_embeddings: int
    rope_type: Literal["yarn"]
    truncate: bool = True
    type: Literal["yarn"] | None = None
    attention_factor: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None


@dataclass(frozen=True)
class DFlashRopeParameters:
    rope_theta: float
    rope_type: Literal["default"]


@dataclass(frozen=True)
class HFDFlashInnerConfig:
    mask_token_id: int
    target_layer_ids: tuple[int, ...]


@dataclass(frozen=True)
class _HFMuseGlimmerAssistantConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    architectures: tuple[Literal["MuseGlimmerAssistantModel"], ...]
    attention_dropout: Literal[0]
    block_size: int
    bos_token_id: int
    dtype: Literal["bfloat16"]
    eos_token_id: int
    head_dim: int
    hidden_act: Literal["silu"]
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[Literal["sliding_attention"], ...]
    mask_token_id: int
    max_position_embeddings: int
    model_type: Literal["muse_glimmer_assistant"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    rms_norm_eps: float
    rope_parameters: DFlashRopeParameters
    sliding_window: int
    target_layer_ids: tuple[int, ...]
    transformers_version: str

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> Self:
        return cls._converter.structure(config, cls)

    def to_qwen_dflash_config(self) -> "HFDFlashConfig":
        return HFDFlashConfig(
            architectures=("DFlashDraftModel",),
            model_type="qwen3",
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_parameters.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
            tie_word_embeddings=False,
            attention_bias=False,
            block_size=self.block_size,
            num_target_layers=52,
            vocab_size=202_048,
            dflash_config=HFDFlashInnerConfig(
                mask_token_id=self.mask_token_id,
                target_layer_ids=self.target_layer_ids,
            ),
            head_dim=self.head_dim,
            layer_types=self.layer_types,
            sliding_window=2 * self.sliding_window,
            use_sliding_window=True,
            sliding_attention_is_causal=False,
        )


@dataclass(frozen=True)
class HFDFlashConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    architectures: tuple[Literal["DFlashDraftModel"], ...]
    model_type: Literal["qwen3"]
    hidden_act: Literal["silu"]
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    block_size: int
    num_target_layers: int
    vocab_size: int
    dflash_config: HFDFlashInnerConfig
    head_dim: int
    layer_types: tuple[Literal["full_attention", "sliding_attention"], ...]
    sliding_window: int | None
    use_sliding_window: bool
    rope_scaling: DFlashYarnRopeScalingConfig | None = None
    sliding_attention_is_causal: bool = True

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with json_path.open() as config_file:
            return cls.from_dict(json.load(config_file))

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> Self:
        if config.get("model_type") == "muse_glimmer_assistant":
            return _HFMuseGlimmerAssistantConfig.from_dict(config).to_qwen_dflash_config()

        config = dict(config)
        dflash_inner = config.get("dflash_config")
        if isinstance(dflash_inner, dict) and "block_size" in dflash_inner:
            dflash_inner = dict(dflash_inner)
            config["block_size"] = dflash_inner.pop("block_size")
            config["dflash_config"] = dflash_inner
        rope_parameters = config.pop("rope_parameters", None)
        if rope_parameters is not None:
            config["rope_theta"] = cls._converter.structure(rope_parameters, DFlashRopeParameters).rope_theta
        return cls._converter.structure(config, cls)

    def _rope_config(self, max_sequence_length: int) -> RoPEConfig:
        if self.rope_scaling is None:
            return UnscaledRoPEConfig(
                base=self.rope_theta,
                max_sequence_length=max_sequence_length,
                head_dim=self.head_dim,
            )
        return YARNRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=max_sequence_length,
            scaling_factor=self.rope_scaling.factor,
            original_context_length=self.rope_scaling.original_max_position_embeddings,
            beta_fast=self.rope_scaling.beta_fast,
            beta_slow=self.rope_scaling.beta_slow,
            truncate=self.rope_scaling.truncate,
            head_dim=self.head_dim,
        )

    def _attention_scale(self) -> float:
        scale = self.head_dim**-0.5
        if self.rope_scaling is None:
            return scale
        scaling = self.rope_scaling
        attention_factor = scaling.attention_factor
        if attention_factor is None and scaling.mscale and scaling.mscale_all_dim:
            if scaling.factor > 1:
                log_factor = math.log(scaling.factor)
                attention_factor = (0.1 * scaling.mscale * log_factor + 1.0) / (
                    0.1 * scaling.mscale_all_dim * log_factor + 1.0
                )
            else:
                attention_factor = 1.0
        if attention_factor is None:
            return scale
        yarn_attention_factor = 0.1 * math.log(scaling.factor) + 1.0
        return scale * (attention_factor / yarn_attention_factor) ** 2

    def _layer_sliding_window_sizes(self) -> tuple[int | None, ...]:
        assert len(self.layer_types) == self.num_hidden_layers

        has_sliding_attention = any(layer_type == "sliding_attention" for layer_type in self.layer_types)
        if has_sliding_attention and not self.use_sliding_window:
            raise ValueError("DFlash config has sliding_attention layers but use_sliding_window is false")
        if self.use_sliding_window and not has_sliding_attention:
            raise ValueError("DFlash config enables sliding windows but has no sliding_attention layers")
        if has_sliding_attention and self.sliding_window is None:
            raise ValueError("DFlash config has sliding_attention layers but does not define sliding_window")
        if not has_sliding_attention and self.sliding_window is not None:
            raise ValueError("DFlash config defines sliding_window without sliding_attention layers")
        return tuple(
            self.sliding_window if layer_type == "sliding_attention" else None for layer_type in self.layer_types
        )

    def to_dflash_draft_config(self, context_length: int | None = None) -> DFlashDraftConfig:
        assert self.dflash_config.target_layer_ids
        assert all(0 <= layer_id < self.num_target_layers for layer_id in self.dflash_config.target_layer_ids)

        max_sequence_length = self.max_position_embeddings if context_length is None else context_length
        linear_config = LinearConfig()
        norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            gate_clipping=None,
            up_clipping=None,
        )
        rope_config = self._rope_config(max_sequence_length)
        layer_configs = tuple(
            TransformerLayerConfig(
                pre_mixer_norm_config=norm_config,
                mixer_config=AttentionConfig(
                    qkvg_projection_config=linear_config,
                    out_projection_config=linear_config,
                    query_norm_config=norm_config,
                    key_norm_config=norm_config,
                    num_heads=self.num_attention_heads,
                    num_groups=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    is_causal=sliding_window_size is not None and self.sliding_attention_is_causal,
                    scale=self._attention_scale(),
                    sliding_window_size=sliding_window_size,
                    logit_soft_cap=None,
                    has_sinks=False,
                    has_qkvg_biases=self.attention_bias,
                    has_out_biases=self.attention_bias,
                ),
                post_mixer_norm_config=None,
                pre_mlp_norm_config=norm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
                rope_config=rope_config,
            )
            for sliding_window_size in self._layer_sliding_window_sizes()
        )
        return DFlashDraftConfig(
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            block_size=self.block_size,
            mask_token_id=self.dflash_config.mask_token_id,
            target_layer_ids=self.dflash_config.target_layer_ids,
            num_target_layers=self.num_target_layers,
            vocab_size=self.vocab_size,
            context_projection_config=linear_config,
            context_norm_config=norm_config,
            rope_config=rope_config,
            layer_configs=layer_configs,
            output_norm_config=norm_config,
        )
