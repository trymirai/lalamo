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
from lalamo.modules.speculators.dflash import (
    DFlashDraftConfig,
    DFlashDraftLayerConfig,
)
from lalamo.modules.token_mixers.attention import AttentionConfig

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

    def resolved_attention_factor(self) -> float | None:
        if self.attention_factor is not None:
            return self.attention_factor
        if not self.mscale or not self.mscale_all_dim:
            return None
        if self.factor <= 1:
            return 1.0
        log_factor = math.log(self.factor)
        return (0.1 * self.mscale * log_factor + 1.0) / (0.1 * self.mscale_all_dim * log_factor + 1.0)


@dataclass(frozen=True)
class DFlashRopeParameters:
    rope_theta: float
    rope_type: Literal["default"]


@dataclass(frozen=True)
class HFDFlashInnerConfig:
    mask_token_id: int
    target_layer_ids: tuple[int, ...]


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

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with json_path.open() as config_file:
            return cls.from_dict(json.load(config_file))

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> Self:
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
        attention_factor = self.rope_scaling.resolved_attention_factor()
        if attention_factor is None:
            return scale
        yarn_attention_factor = 0.1 * math.log(self.rope_scaling.factor) + 1.0
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
            DFlashDraftLayerConfig(
                attention_config=AttentionConfig(
                    qkv_projection_config=linear_config,
                    out_projection_config=linear_config,
                    query_norm_config=norm_config,
                    key_norm_config=norm_config,
                    num_heads=self.num_attention_heads,
                    num_groups=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    is_causal=False,
                    scale=self._attention_scale(),
                    sliding_window_size=2 * sliding_window_size - 1 if sliding_window_size is not None else None,
                    logit_soft_cap=None,
                    has_sinks=False,
                    has_qkv_biases=self.attention_bias,
                    has_out_biases=self.attention_bias,
                ),
                input_norm_config=norm_config,
                post_attention_norm_config=norm_config,
                mlp_config=mlp_config,
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
