import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import cattrs

__all__ = ["RopeScalingConfig", "LlamaConfig"]


@dataclass
class RopeScalingConfig:
    factor: float
    high_freq_factor: float
    low_freq_factor: float
    original_max_position_embeddings: int
    rope_type: Literal["llama3"]


@dataclass
class LlamaConfig:
    architectures: list[Literal["LlamaForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    head_dim: int
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["llama"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_scaling: RopeScalingConfig
    rope_theta: float
    tie_word_embeddings: bool
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    transformers_version: str
    use_cache: bool
    vocab_size: int

    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    @classmethod
    def from_json(cls, json_path: Path | str) -> "LlamaConfig":
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)
