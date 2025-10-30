from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

import cattrs
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.model_import.decoder_configs import ForeignConfig
from lalamo.model_import.loaders import load_huggingface
from lalamo.modules import Decoder

__all__ = [
    "AWQQuantizationConfig",
    "GPTQMetaConfig",
    "GPTQQuantizationConfig",
    "HuggingFaceConfig",
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
class MLXQuantizationConfig:
    group_size: int
    bits: int

QuantizationConfigType = AWQQuantizationConfig | GPTQQuantizationConfig | MLXQuantizationConfig | None

def _structure_quantization_config(v: object, _: object) -> QuantizationConfigType:
    match v:
        case None:
            return None

        case {"quant_method": "awq", **_other}:
            return cattrs.structure(v, AWQQuantizationConfig)

        case {"quant_method": "gptq", **_other}:
            return cattrs.structure(v, GPTQQuantizationConfig)

        case {**_other}:
            return cattrs.structure(v, MLXQuantizationConfig)

        case _:
            raise RuntimeError(f"Cannot structure {v}field")

@dataclass(frozen=True)
class HuggingFaceConfig(ForeignConfig):
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)
    _converter.register_structure_hook(QuantizationConfigType, _structure_quantization_config)

    @property
    def eos_token_ids(self) -> list[int]:
        if not hasattr(self, "eos_token_id"):
            raise RuntimeError("model doesn't havve eos_token_id, override eos_token_ids in model config")

        return [self.eos_token_id] if isinstance(self.eos_token_id, int) else self.eos_token_id

    @property
    def default_precision(self) -> DTypeLike:
        return jnp.dtype(getattr(self, "torch_dtype", "bfloat16"))

    @classmethod
    def _load_weights(
        cls,
        model: Decoder,
        weights_dict: Mapping[str, Array],
    ) -> Decoder:
        return load_huggingface(model, weights_dict)
