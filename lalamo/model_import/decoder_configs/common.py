import json
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
from jaxtyping import Array, DTypeLike

from lalamo.modules import ClassifierConfig, DecoderConfig
from lalamo.modules.common import LalamoModule
from lalamo.registry_abc import RegistryABC

__all__ = ["ForeignClassifierConfig", "ForeignLMConfig"]


@dataclass(frozen=True)
class ForeignConfig[ConfigT: DecoderConfig | ClassifierConfig](RegistryABC):
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    @property
    @abstractmethod
    def default_precision(self) -> DTypeLike: ...

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
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
