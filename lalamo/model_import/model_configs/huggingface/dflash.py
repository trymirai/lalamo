import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Self

import cattrs

from lalamo.modules.activations import SiLU
from lalamo.modules.dflash import DFlashDraftConfig, make_qwen3_dflash_draft_config
from lalamo.modules.rope import RoPEConfig, UnscaledRoPEConfig, YARNRoPEConfig

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
    truncate: bool
    type: Literal["yarn"] | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None


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
    rope_scaling: DFlashYarnRopeScalingConfig | None
    layer_types: tuple[Literal["full_attention", "sliding_attention"], ...]
    sliding_window: int | None
    use_sliding_window: bool

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with json_path.open() as config_file:
            return cls.from_dict(json.load(config_file))

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> Self:
        return cls._converter.structure(config, cls)

    def _target_layer_ids(self) -> tuple[int, ...]:
        if not self.dflash_config.target_layer_ids:
            raise ValueError("DFlash config defines empty target_layer_ids")
        invalid_layer_ids = tuple(
            layer_id
            for layer_id in self.dflash_config.target_layer_ids
            if layer_id < 0 or layer_id >= self.num_target_layers
        )
        if invalid_layer_ids:
            raise ValueError(
                "DFlash config target_layer_ids must be within target model layers:"
                f" invalid={invalid_layer_ids}, num_target_layers={self.num_target_layers}",
            )
        return self.dflash_config.target_layer_ids

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

    def _layer_types(self) -> tuple[Literal["full_attention", "sliding_attention"], ...]:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"Expected {self.num_hidden_layers} layer_types entries, got {len(self.layer_types)}",
            )
        return self.layer_types

    def _layer_sliding_window_sizes(self) -> tuple[int | None, ...]:
        layer_types = self._layer_types()
        has_sliding_attention = any(layer_type == "sliding_attention" for layer_type in layer_types)
        if has_sliding_attention and not self.use_sliding_window:
            raise ValueError("DFlash config has sliding_attention layers but use_sliding_window is false")
        if self.use_sliding_window and not has_sliding_attention:
            raise ValueError("DFlash config enables sliding windows but has no sliding_attention layers")
        if has_sliding_attention and self.sliding_window is None:
            raise ValueError("DFlash config has sliding_attention layers but does not define sliding_window")
        if not has_sliding_attention and self.sliding_window is not None:
            raise ValueError("DFlash config defines sliding_window without sliding_attention layers")
        return tuple(self.sliding_window if layer_type == "sliding_attention" else None for layer_type in layer_types)

    def to_dflash_draft_config(self, context_length: int | None = None) -> DFlashDraftConfig:
        max_sequence_length = self.max_position_embeddings if context_length is None else context_length
        return make_qwen3_dflash_draft_config(
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            block_size=self.block_size,
            mask_token_id=self.dflash_config.mask_token_id,
            target_layer_ids=self._target_layer_ids(),
            num_target_layers=self.num_target_layers,
            vocab_size=self.vocab_size,
            num_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_layers=self.num_hidden_layers,
            rms_norm_eps=self.rms_norm_eps,
            rope_config=self._rope_config(max_sequence_length),
            layer_sliding_window_sizes=self._layer_sliding_window_sizes(),
            attention_bias=self.attention_bias,
            activation=SiLU(),
        )
