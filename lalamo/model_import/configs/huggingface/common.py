from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.model_import.configs import ForeignConfig
from lalamo.model_import.loaders import load_huggingface
from lalamo.modules import Decoder

__all__ = [
    "HuggingFaceConfig",
    "AWQQuantizationConfig",
    "GPTQMetaConfig",
    "GPTQQuantizationConfig"
]


@dataclass(frozen=True)
class AWQQuantizationConfig:
    backend: Literal["autoawq"] = "autoawq"
    bits: Literal[4, 8] = 4
    do_fuse: Literal[False] = False
    exllama_config: None = None
    fuse_max_seq_len: None = None
    group_size: int = 128
    modules_to_fuse: None = None
    modules_to_not_convert: None = None
    quant_method: Literal["awq"] = "awq"
    version: Literal["gemm"] = "gemm"
    zero_point: bool = True


@dataclass(frozen=True)
class GPTQMetaConfig:
    damp_auto_increment: float
    damp_percent: float
    mse: float
    quantizer: list[str]
    static_groups: bool
    true_sequential: bool
    uri: str


@dataclass(frozen=True)
class GPTQQuantizationConfig:
    bits: int
    checkpoint_format: str
    desc_act: bool
    group_size: int
    lm_head: bool
    meta: GPTQMetaConfig
    pack_dtype: str
    quant_method: Literal["gptq"]
    sym: bool


@dataclass(frozen=True)
class HuggingFaceConfig(ForeignConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]

    @property
    def default_precision(self) -> DTypeLike:
        return jnp.dtype(self.torch_dtype)

    @classmethod
    def _load_weights(
        cls,
        model: Decoder,
        weights_dict: dict[str, Array],
    ) -> Decoder:
        return load_huggingface(model, weights_dict)
