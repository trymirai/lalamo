from dataclasses import dataclass, field
from typing import ClassVar, Self

from cattrs import GenConverter

from lalamo.model_import.model_configs.foreign_config import (
    ForeignClassifierConfig,
    ForeignConfig,
    ForeignLMConfig,
    ForeignTTSConfig,
)
from lalamo.model_import.origins import FileSpec, Origin
from lalamo.models.language_model import GenerationConfig
from lalamo.utils.json import JSON
from lalamo.utils.registry_abc import RegistryABC, make_registry_abc_converter

__all__ = [
    "ClassifierModelSpec",
    "ConfigMap",
    "FileSpec",
    "JSONFieldSpec",
    "LanguageModelSpec",
    "ModelSpec",
    "TTSModelSpec",
]


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
    generation_params_overrides: GenerationConfig | None = None
    chat_template: FileSpec | JSONFieldSpec | str | None = None
    system_prompt: FileSpec | str | None = None


@dataclass(frozen=True)
class ModelSpec[ForeignConfigBaseT: ForeignConfig](RegistryABC):
    _converter: ClassVar[GenConverter] = make_registry_abc_converter()

    vendor: str
    family: str
    name: str
    size: str
    origin: Origin
    config_type: type[ForeignConfigBaseT]
    configs: ConfigMap = field(default=ConfigMap())

    def __post_init__(self) -> None:
        if self.__class__ is ModelSpec:
            raise TypeError("ModelSpec is abstract; instantiate a concrete model spec type.")

    @classmethod
    def from_json(cls, json_object: JSON) -> Self:
        return cls._converter.structure(json_object, cls)

    def to_json(self) -> JSON:
        return self._converter.unstructure(self)


@dataclass(frozen=True)
class LanguageModelSpec(ModelSpec[ForeignLMConfig]):
    config_type: type[ForeignLMConfig]
    output_parser_regex: str | None = None
    end_of_thinking_tag: str | None = None
    system_role_name: str = "system"
    user_role_name: str = "user"
    assistant_role_name: str = "assistant"
    grammar_start_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassifierModelSpec(ModelSpec[ForeignClassifierConfig]):
    config_type: type[ForeignClassifierConfig]
    output_parser_regex: str | None = None
    system_role_name: str = "system"
    user_role_name: str = "user"
    assistant_role_name: str = "assistant"


@dataclass(frozen=True)
class TTSModelSpec(ModelSpec[ForeignTTSConfig]):
    config_type: type[ForeignTTSConfig]
