from collections.abc import (
    Callable,
    Iterator,
    Mapping,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, ClassVar, cast, get_args, get_origin

import cattrs
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.model_import.decoder_configs import ForeignConfig
from lalamo.models.language_model import GenerationConfig
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
    generation_config: FileSpec | GenerationConfig | None = field(default=FileSpec("generation_config.json"))
    chat_template: FileSpec | JSONFieldSpec | str | None = None


def _is_foreign_config_type(t: object) -> bool:
    origin = get_origin(t)
    args = get_args(t)
    return origin is type and len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], ForeignConfig)


def _structure_foreign_config_factory(
    t: object,  # noqa: ARG001
    c: cattrs.Converter,  # noqa: ARG001
) -> Callable[[object, object], type[ForeignConfig]]:
    name_to_type = {t.__name__: t for t in ForeignConfig.__descendants__()}

    def _hook(v: object, _t: object) -> type[ForeignConfig]:
        if isinstance(v, type) and issubclass(v, ForeignConfig):
            return v
        return name_to_type[cast("str", v)]

    return _hook


def _unstructure_foreign_config_factory(t: object, c: cattrs.Converter) -> Callable[[type[ForeignConfig]], str]:  # noqa: ARG001
    def _hook(v: type[ForeignConfig]) -> str:
        return v.__name__

    return _hook


def _structure_chat_template(value: object, _type: object) -> FileSpec | JSONFieldSpec | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value = cast("dict[Any, Any]", value) # ty bug??? Why is just `dict` != `dict[Any, Any]`?
        if "file_spec" in value and "field_name" in value:
            return JSONFieldSpec(
                file_spec=FileSpec(**value["file_spec"]),
                field_name=value["field_name"],
            )
        if "filename" in value:
            return FileSpec(**value)
    raise ValueError(f"Invalid chat_template value: {value}")


@dataclass(frozen=True)
class ModelSpec:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    _converter.register_structure_hook_factory(_is_foreign_config_type, _structure_foreign_config_factory)
    _converter.register_unstructure_hook_factory(_is_foreign_config_type, _unstructure_foreign_config_factory)
    _converter.register_structure_hook(FileSpec | JSONFieldSpec | str | None, _structure_chat_template)

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
