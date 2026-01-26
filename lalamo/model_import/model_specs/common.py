from collections.abc import (
    Callable,
    Iterator,
    Mapping,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import ClassVar, get_args, get_origin

import jax.numpy as jnp
from cattrs import GenConverter
from cattrs.preconf.json import make_converter
from jaxtyping import Array, DTypeLike

from lalamo.common import JSON
from lalamo.model_import.decoder_configs import ForeignConfig
from lalamo.quantization import QuantizationMode
from lalamo.safetensors import safe_read
from lalamo.utils import MapDictValues

__all__ = [
    "ConfigMap",
    "FileSpec",
    "JSONFieldSpec",
    "ModelSpec",
    "ModelType",
    "UseCase",
    "WeightsType",
    "awq_model_spec",
    "build_quantized_models",
]


class ModelType(StrEnum):
    LANGUAGE_MODEL = "language_model"
    CLASSIFIER_MODEL = "classifier_model"


def cast_if_float(array: Array, cast_to: DTypeLike) -> Array:
    if array.dtype in [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]:
        return array.astype(cast_to)
    return array


class WeightsType(Enum):
    SAFETENSORS = "safetensors"
    TORCH = "torch"

    @contextmanager
    def load(
        self,
        filename: Path | str,
        float_dtype: DTypeLike,
    ) -> Iterator[tuple[Mapping[str, jnp.ndarray], Mapping[str, str]]]:
        if self == WeightsType.SAFETENSORS:
            with Path(filename).open("rb") as fd:
                (metadata_dict, weights_dict) = safe_read(fd)
                yield MapDictValues(lambda v: cast_if_float(v, float_dtype), weights_dict), metadata_dict or {}
        else:
            import torch

            from lalamo.modules.torch_interop import torch_to_jax

            torch_weights = torch.load(filename, map_location="cpu", weights_only=True)
            yield MapDictValues(lambda v: cast_if_float(torch_to_jax(v), float_dtype), torch_weights), {}


class UseCase(Enum):
    CODE = "code"


@dataclass(frozen=True)
class FileSpec:
    filename: str
    repo: str | None = None


@dataclass(frozen=True)
class JSONFieldSpec:
    file_spec: FileSpec
    field_name: str


@dataclass(frozen=True)
class ConfigMap:
    model_config: FileSpec = field(default=FileSpec("config.json"))
    tokenizer: FileSpec = field(default=FileSpec("tokenizer.json"))
    tokenizer_config: FileSpec = field(default=FileSpec("tokenizer_config.json"))
    generation_config: FileSpec | None = field(default=FileSpec("generation_config.json"))
    chat_template: FileSpec | JSONFieldSpec | str | None = None


ChatTemplateType = FileSpec | JSONFieldSpec | str | None


def _is_foreign_config_type(t: type) -> bool:
    origin = get_origin(t)
    args = get_args(t)
    return origin is type and len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], ForeignConfig)


def _is_chat_template_type(t: type) -> bool:
    return t == ChatTemplateType


def _make_model_spec_converter() -> GenConverter:
    converter = make_converter()

    @converter.register_unstructure_hook_factory(_is_foreign_config_type)
    def unstructure_foreign_config_type(
        t: type,  # noqa: ARG001
        c: GenConverter,  # noqa: ARG001
    ) -> Callable[[type[ForeignConfig]], str]:
        def unstructure(v: type[ForeignConfig]) -> str:
            return v.__name__

        return unstructure

    @converter.register_structure_hook_factory(_is_foreign_config_type)
    def structure_foreign_config_type(
        t: type,  # noqa: ARG001
    ) -> Callable[[str | type[ForeignConfig], type], type[ForeignConfig]]:
        name_to_type: dict[str, type[ForeignConfig]] = {cls.__name__: cls for cls in ForeignConfig.__descendants__()}

        def structure(v: str | type[ForeignConfig], _t: type) -> type[ForeignConfig]:
            if isinstance(v, type) and issubclass(v, ForeignConfig):
                return v
            return name_to_type[v]

        return structure

    @converter.register_structure_hook_factory(_is_chat_template_type)
    def structure_chat_template(
        t: type,  # noqa: ARG001
    ) -> Callable[[JSON, type], FileSpec | JSONFieldSpec | str | None]:
        def structure(value: JSON, _t: type) -> FileSpec | JSONFieldSpec | str | None:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                if "file_spec" in value and "field_name" in value:
                    return converter.structure(value, JSONFieldSpec)
                if "filename" in value:
                    return converter.structure(value, FileSpec)
            raise ValueError(f"Invalid chat_template value: {value}")

        return structure

    return converter


@dataclass(frozen=True)
class ModelSpec:
    _converter: ClassVar[GenConverter] = _make_model_spec_converter()

    vendor: str
    family: str
    name: str
    size: str
    repo: str
    config_type: type[ForeignConfig]
    quantization: QuantizationMode | None = None
    output_parser_regex: str | None = None
    system_role_name: str = "system"
    user_role_name: str = "user"
    assistant_role_name: str = "assistant"
    tool_role_name: str = "tool"
    weights_type: WeightsType = WeightsType.SAFETENSORS
    model_type: ModelType = ModelType.LANGUAGE_MODEL
    configs: ConfigMap = field(default=ConfigMap())
    use_cases: tuple[UseCase, ...] = tuple()
    grammar_start_tokens: tuple[str, ...] = tuple()

    @classmethod
    def from_json(cls, json_data: dict) -> "ModelSpec":
        return cls._converter.structure(json_data, cls)

    def to_json(self) -> dict:
        return self._converter.unstructure(self)


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
        grammar_start_tokens=model_spec.grammar_start_tokens,
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
