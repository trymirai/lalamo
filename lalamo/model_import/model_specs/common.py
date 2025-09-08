from collections.abc import (
    Mapping,
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike
from safetensors.flax import load_file as load_safetensors

from lalamo.model_import.decoder_configs import ForeignConfig
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.quantization import QuantizationMode
from lalamo.utils import MapDictValues

__all__ = [
    "ConfigMap",
    "FileSpec",
    "ModelSpec",
    "UseCase",
    "WeightsType",
    "awq_model_spec",
    "build_quantized_models",
]


def cast_if_float(array: Array, cast_to: DTypeLike) -> Array:
    if array.dtype in [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]:
        return array.astype(cast_to)
    return array


class WeightsType(Enum):
    SAFETENSORS = "safetensors"
    TORCH = "torch"

    def load(self, filename: Path | str, float_dtype: DTypeLike) -> Mapping[str, jnp.ndarray]:
        if self == WeightsType.SAFETENSORS:
            return MapDictValues(lambda v: cast_if_float(v, float_dtype), load_safetensors(filename))

        import torch

        torch_weights = torch.load(filename, map_location="cpu", weights_only=True)
        return MapDictValues(lambda v: cast_if_float(torch_to_jax(v), float_dtype), torch_weights)


class UseCase(Enum):
    CODE = "code"


@dataclass(frozen=True)
class FileSpec:
    filename: str
    repo: str | None = None


@dataclass(frozen=True)
class ConfigMap:
    model_config: FileSpec = field(default=FileSpec("config.json"))
    tokenizer: FileSpec = field(default=FileSpec("tokenizer.json"))
    tokenizer_config: FileSpec = field(default=FileSpec("tokenizer_config.json"))
    generation_config: FileSpec | None = field(default=FileSpec("generation_config.json"))
    chat_template: FileSpec | None = None


@dataclass(frozen=True)
class ModelSpec:
    vendor: str
    family: str
    name: str
    size: str
    quantization: QuantizationMode | None
    repo: str
    config_type: type[ForeignConfig]
    output_parser_regex: str | None = None
    system_role_name: str = "system"
    user_role_name: str = "user"
    assistant_role_name: str = "assistant"
    tool_role_name: str = "tool"
    weights_type: WeightsType = WeightsType.SAFETENSORS
    configs: ConfigMap = field(default=ConfigMap())
    use_cases: tuple[UseCase, ...] = tuple()


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
        configs=model_spec.configs,
        weights_type=model_spec.weights_type,
        use_cases=model_spec.use_cases,
    )


def build_quantized_models(model_specs: list[ModelSpec]) -> list[ModelSpec]:
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
