import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cattrs

__all__ = ["LlamaConfig"]


@dataclass
class QuantizationConfig:
    group_size: int


@dataclass
class LoraConfig:
    rank: int
    scale: float


@dataclass
class LlamaConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    ffn_dim_multiplier: float
    multiple_of: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    quantization_args: QuantizationConfig | None = None
    lora_args: LoraConfig | None = None

    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    @classmethod
    def from_json(cls, json_path: Path | str) -> "LlamaConfig":
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)
