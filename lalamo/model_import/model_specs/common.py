import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, ClassVar, cast, get_args, get_origin

import cattrs

from lalamo.model_import.model_configs import ForeignConfig
from lalamo.models.language_model import GenerationConfig
from lalamo.quantization import QuantizationMode

from .origins import FileSpec, HuggingFaceOrigin, Origin

__all__ = [
    "ConfigMap",
    "FileSpec",
    "JSONFieldSpec",
    "ModelSpec",
    "ModelType",
    "UseCase",
    "awq_model_spec",
    "build_quantized_models",
    "structure_origin",
]


class ModelType(StrEnum):
    LANGUAGE_MODEL = "language_model"
    CLASSIFIER_MODEL = "classifier_model"
    TTS_MODEL = "tts_model"
    LATENT_TTS_MODEL = "latent_tts_model"


class UseCase(Enum):
    CODE = "code"


@dataclass(frozen=True)
class JSONFieldSpec:
    file_spec: FileSpec
    field_name: str


@dataclass(frozen=True)
class ConfigMap:
    model_config: FileSpec = field(default=FileSpec("config.json"))
    tokenizer: FileSpec | None = field(default=FileSpec("tokenizer.json"))
    tokenizer_config: FileSpec = field(default=FileSpec("tokenizer_config.json"))
    generation_config: FileSpec | GenerationConfig | None = field(default=FileSpec("generation_config.json"))
    chat_template: FileSpec | JSONFieldSpec | str | None = None
    system_prompt: FileSpec | str | None = None
    extra_configs: tuple[FileSpec, ...] = ()


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


def _structure_system_prompt(value: object, _type: object) -> FileSpec | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value = cast("dict[Any, Any]", value)
        if "filename" in value:
            return FileSpec(**value)
    raise ValueError(f"Invalid system_prompt value: {value}")


def _structure_chat_template(value: object, _type: object) -> FileSpec | JSONFieldSpec | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value = cast("dict[Any, Any]", value)  # ty bug??? Why is just `dict` != `dict[Any, Any]`?
        if "file_spec" in value and "field_name" in value:
            return JSONFieldSpec(
                file_spec=FileSpec(**value["file_spec"]),
                field_name=value["field_name"],
            )
        if "filename" in value:
            return FileSpec(**value)
    raise ValueError(f"Invalid chat_template value: {value}")


def structure_origin(data: dict[str, Any] | Origin) -> Origin:
    if isinstance(data, Origin):
        return data
    data = dict(data)
    type_name = data.pop("type")
    name_to_type = {t.__name__: t for t in Origin.__descendants__()}
    origin_type = name_to_type.get(type_name)
    if origin_type is None:
        available = ", ".join(sorted(name_to_type))
        raise ValueError(f"Unknown origin type: {type_name!r}. Available: {available}")
    return cattrs.structure(data, origin_type)


def _unstructure_origin(obj: Origin) -> dict:
    fields: dict[str, Any] = {}
    for f in dataclasses.fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, Enum):
            value = value.value
        fields[f.name] = value
    return {"type": type(obj).__name__, **fields}


@dataclass(frozen=True)
class ModelSpec:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    _converter.register_structure_hook_factory(_is_foreign_config_type, _structure_foreign_config_factory)
    _converter.register_unstructure_hook_factory(_is_foreign_config_type, _unstructure_foreign_config_factory)
    _converter.register_structure_hook(FileSpec | JSONFieldSpec | str | None, _structure_chat_template)
    _converter.register_structure_hook(FileSpec | str | None, _structure_system_prompt)
    _converter.register_structure_hook(Origin, lambda data, _type: structure_origin(data))
    _converter.register_unstructure_hook(Origin, _unstructure_origin)

    vendor: str
    family: str
    name: str
    size: str
    origin: Origin
    config_type: type[ForeignConfig]
    quantization: QuantizationMode | None = None
    output_parser_regex: str | None = None
    system_role_name: str = "system"
    user_role_name: str = "user"
    assistant_role_name: str = "assistant"
    tool_role_name: str = "tool"
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
    origin: Origin,
    quantization: QuantizationMode = QuantizationMode.UINT4,
) -> ModelSpec:
    return ModelSpec(
        vendor=model_spec.vendor,
        family=model_spec.family,
        name=f"{model_spec.name}-AWQ",
        size=model_spec.size,
        quantization=quantization,
        origin=origin,
        config_type=model_spec.config_type,
        configs=model_spec.configs,
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
        if not isinstance(model_spec.origin, HuggingFaceOrigin):
            continue
        if model_spec.origin.repo not in quantization_compatible_repos:
            continue
        quantized_origin = HuggingFaceOrigin(repo="trymirai/{}-AWQ".format(model_spec.origin.repo.split("/")[-1]))
        quantized_model_spec = awq_model_spec(model_spec, quantized_origin)
        quantized_model_specs.append(quantized_model_spec)
    return quantized_model_specs
