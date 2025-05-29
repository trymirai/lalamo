from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
import torch
from jaxtyping import Array
from safetensors.flax import load_file as load_safetensors

from fartsovka.common import DType
from fartsovka.model_import.configs import ForeignConfig
from fartsovka.quantization import QuantizationMode

__all__ = [
    "HUGGINGFACE_TOKENIZER_FILES",
    "ModelSpec",
    "huggingface_weight_files",
]


@torch.no_grad()
def _torch_to_jax_bfloat16(tensor: torch.Tensor) -> Array:
    # Credit: https://github.com/jax-ml/ml_dtypes/issues/81#issuecomment-2399636232
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Trying to convert non-bfloat16 tensor to bfloat16")
    intermediate_tensor = tensor.view(torch.uint16)
    return jnp.array(intermediate_tensor).view("bfloat16")


def _convert_torch_array(array: torch.Tensor, float_dtype: DType) -> Array:
    array = array.detach().cpu()
    if array.dtype == torch.bfloat16:
        jax_array = _torch_to_jax_bfloat16(array)
    else:
        jax_array = jnp.array(array.numpy())
    return jax_array.astype(float_dtype)


class WeightsType(Enum):
    SAFETENSORS = "safetensors"
    TORCH = "torch"

    def load(self, filename: Path | str, float_dtype: DType) -> dict[str, jnp.ndarray]:
        if self == WeightsType.SAFETENSORS:
            return {k: v.astype(float_dtype) for k, v in load_safetensors(filename).items()}
        torch_weights = torch.load(filename, map_location="cpu", weights_only=True)
        return {k: _convert_torch_array(v, float_dtype) for k, v in torch_weights.items()}


@dataclass
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


def huggingface_weight_files(num_shards: int) -> tuple[str, ...]:
    if num_shards == 1:
        return ("model.safetensors",)
    return tuple(f"model-{i:05d}-of-{num_shards:05d}.safetensors" for i in range(1, num_shards + 1))


HUGGINGFACE_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
