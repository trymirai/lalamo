import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, TypeVar

import cattrs

__all__ = ["RopeScalingConfig", "LlamaConfig", "Qwen2Config", "GemmaConfig"]


T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    @classmethod
    def from_json(cls: type[T], json_path: Path | str) -> T:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)

    def to_json(self, json_path: Path | str) -> None:
        json_path = Path(json_path)
        with open(json_path, "w") as f:
            json.dump(self._converter.unstructure(self), f, indent=2)


@dataclass
class RopeScalingConfig:
    factor: float
    high_freq_factor: float
    low_freq_factor: float
    original_max_position_embeddings: int
    rope_type: Literal["llama3"]


@dataclass
class LlamaConfig(BaseConfig):
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


@dataclass
class Qwen2Config(BaseConfig):
    architectures: list[Literal["Qwen2ForCausalLM"]]
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    model_type: Literal["qwen2"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    tie_word_embeddings: bool
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int


@dataclass
class GemmaConfig(BaseConfig):
    architectures: list[Literal["Gemma2ForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    attn_logit_softcapping: float
    bos_token_id: int | list[int]
    cache_implementation: Literal["hybrid"]
    eos_token_id: int | list[int]
    final_logit_softcapping: float
    head_dim: int
    hidden_act: Literal["gelu_pytorch_tanh"]
    hidden_activation: Literal["gelu_pytorch_tanh"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: Literal["gemma2"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    query_pre_attn_scalar: float
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    transformers_version: str
    use_cache: bool
    vocab_size: int
