import json
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
from jaxtyping import Array, DTypeLike

from lalamo.modules import ClassifierConfig, DecoderConfig, TTSConfig
from lalamo.modules.common import LalamoModule
from lalamo.registry_abc import RegistryABC

__all__ = ["ForeignClassifierConfig", "ForeignLMConfig"]

SUPPORTED_CONFIG_TYPES = DecoderConfig | ClassifierConfig | TTSConfig


@dataclass(frozen=True)
class ForeignConfig[ConfigT: SUPPORTED_CONFIG_TYPES](RegistryABC):
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    @property
    @abstractmethod
    def default_precision(self) -> DTypeLike: ...

    @staticmethod
    def _read_and_merge_configs(
        json_path: Path,
        extra_config_paths: Sequence[Path],
    ) -> dict:
        with open(json_path) as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise TypeError(f"Config at {json_path} must be a JSON object, got {type(config).__name__}")

        for extra_path in extra_config_paths:
            with open(extra_path) as f:
                extra = json.load(f)
            if not isinstance(extra, dict):
                raise TypeError(
                    f"Extra config at {extra_path} must be a JSON object, got {type(extra).__name__}",
                )
            merged = dict(extra)
            merged.update(config)
            config = merged

        return config

    @classmethod
    def from_json(cls, json_path: Path | str, extra_config_paths: Sequence[Path] = ()) -> Self:
        config = cls._read_and_merge_configs(Path(json_path), extra_config_paths)
        return cls._converter.structure(config, cls)

    @abstractmethod
    def _load_weights(
        self,
        model: LalamoModule,
        weights_dict: Mapping[str, Array],
    ) -> LalamoModule: ...

    @abstractmethod
    def to_lalamo_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> ConfigT: ...

    def load(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        weights_dict: Mapping[str, Array],
        metadata_dict: Mapping[str, str],
    ) -> LalamoModule[ConfigT]:
        config = self.to_lalamo_config(context_length, activation_precision, accumulation_precision, metadata_dict)
        model = config.empty()
        return self._load_weights(model, weights_dict)


@dataclass(frozen=True)
class ForeignLMConfig(ForeignConfig, RegistryABC):
    @abstractmethod
    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig: ...

    @property
    @abstractmethod
    def eos_token_ids(self) -> list[int]: ...

    def to_lalamo_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig:
        return self.to_decoder_config(context_length, activation_precision, accumulation_precision, metadata_dict)


@dataclass(frozen=True)
class ForeignClassifierConfig(ForeignConfig, RegistryABC):
    @abstractmethod
    def to_classifier_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> ClassifierConfig: ...

    def to_lalamo_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> ClassifierConfig:
        return self.to_classifier_config(context_length, activation_precision, accumulation_precision)


@dataclass(frozen=True)
class ForeignTTSConfig(ForeignConfig, RegistryABC):
    @abstractmethod
    def to_tts_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> TTSConfig: ...

    def to_lalamo_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> TTSConfig:
        return self.to_tts_config(context_length, activation_precision, accumulation_precision)
