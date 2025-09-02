from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike
from safetensors.flax import load_file as load_safetensors

from lalamo.model_import.configs import ForeignConfig
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.quantization import QuantizationMode

__all__ = [
    "HUGGINFACE_GENERATION_CONFIG_FILE",
    "HUGGINGFACE_TOKENIZER_FILES",
    "ModelSpec",
    "TokenizerFileSpec",
    "UseCase",
    "awq_model_spec",
    "build_quantized_models",
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

        import torch

        torch_weights = torch.load(filename, map_location="cpu", weights_only=True)
        return {k: cast_if_float(torch_to_jax(v), float_dtype) for k, v in torch_weights.items()}


class UseCase(Enum):
    CODE = "code"


@dataclass(frozen=True)
class TokenizerFileSpec:
    repo: str | None
    filename: str


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
    tokenizer_files: tuple[TokenizerFileSpec, ...] = tuple()
    use_cases: tuple[UseCase, ...] = tuple()


def huggingface_weight_files(num_shards: int) -> tuple[str, ...]:
    if num_shards == 1:
        return ("model.safetensors",)
    return tuple(f"model-{i:05d}-of-{num_shards:05d}.safetensors" for i in range(1, num_shards + 1))


def awq_model_spec(
    model_spec: ModelSpec,
    repo: str,
    quantization: QuantizationMode = QuantizationMode.UINT4,
) -> ModelSpec:
    return ModelSpec(
        vendor=model_spec.vendor,
        family=model_spec.family,
        name=f"{model_spec.name}-AWQ",
        size=model_spec.size,
        quantization=quantization,
        repo=repo,
        config_type=model_spec.config_type,
        config_file_name=model_spec.config_file_name,
        weights_file_names=huggingface_weight_files(1),
        weights_type=model_spec.weights_type,
        tokenizer_files=model_spec.tokenizer_files,
        use_cases=model_spec.use_cases,
    )


def build_quantized_models(model_specs: list[ModelSpec]):
    quantization_compatible_repos: list[str] = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]

    quantized_model_specs: list[ModelSpec] = []
    for model_spec in model_specs:
        if model_spec.repo not in quantization_compatible_repos:
            continue
        quantized_repo = "trymirai/{}-AWQ".format(model_spec.repo.split("/")[-1])
        quantized_model_spec = awq_model_spec(model_spec, quantized_repo)
        quantized_model_specs.append(quantized_model_spec)
    return quantized_model_specs


HUGGINGFACE_TOKENIZER_FILES = (
    TokenizerFileSpec(repo=None, filename="tokenizer.json"),
    TokenizerFileSpec(repo=None, filename="tokenizer_config.json"),
)

HUGGINFACE_GENERATION_CONFIG_FILE = TokenizerFileSpec(repo=None, filename="generation_config.json")
