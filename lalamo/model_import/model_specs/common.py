from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
import torch
from jaxtyping import Array, DTypeLike
from safetensors.flax import load_file as load_safetensors

from lalamo.model_import.configs import ForeignConfig
from lalamo.quantization import QuantizationMode
from lalamo.utils import torch_to_jax

__all__ = [
    "HUGGINGFACE_TOKENIZER_FILES",
    "ModelSpec",
    "UseCase",
    "huggingface_weight_files",
]


def cast_if_float(array: Array, cast_to: DTypeLike) -> Array:
    if array.dtype in [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]:
        return array.astype(cast_to)
    return array


class WeightsType(Enum):
    SAFETENSORS = "safetensors"
    TORCH = "torch"

    def load(self, filename: Path | str, float_dtype: DTypeLike) -> dict[str, jnp.ndarray]:
        if self == WeightsType.SAFETENSORS:
            return {k: cast_if_float(v, float_dtype) for k, v in load_safetensors(filename).items()}
        torch_weights = torch.load(filename, map_location="cpu", weights_only=True)
        return {k: cast_if_float(torch_to_jax(v), float_dtype) for k, v in torch_weights.items()}


class UseCase(Enum):
    CODE = "code"


@dataclass(frozen=True)
class ModelSpec:
    vendor: str
    family: str
    name: str
    size: str
    quantization: QuantizationMode | None
    repo: str
    config_type: type[ForeignConfig]
    config_file_name: str
    weights_file_names: tuple[str, ...]
    weights_type: WeightsType
    tokenizer_file_names: tuple[str, ...] = tuple()
    use_cases: tuple[UseCase, ...] = tuple()


def huggingface_weight_files(num_shards: int) -> tuple[str, ...]:
    if num_shards == 1:
        return ("model.safetensors",)
    return tuple(f"model-{i:05d}-of-{num_shards:05d}.safetensors" for i in range(1, num_shards + 1))


HUGGINGFACE_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
