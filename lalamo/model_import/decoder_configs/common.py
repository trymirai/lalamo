import json
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
from jaxtyping import Array, DTypeLike

from lalamo.modules import Classifier, ClassifierConfig, Decoder, DecoderConfig
from lalamo.registry_abc import RegistryABC

__all__ = ["ForeignClassifierConfig", "ForeignLMConfig"]


@dataclass(frozen=True)
class ForeignConfig(RegistryABC):
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    eos_token_id: int | list[int]

    @property
    def eos_token_ids(self) -> list[int]:
        return [self.eos_token_id] if isinstance(self.eos_token_id, int) else self.eos_token_id

    @property
    @abstractmethod
    def default_precision(self) -> DTypeLike: ...

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)


@dataclass(frozen=True)
class ForeignLMConfig(ForeignConfig, RegistryABC):
    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        raise NotImplementedError

    @classmethod
    def _load_decoder_weights(
        cls,
        model: Decoder,
        weights_dict: Mapping[str, Array],
    ) -> Decoder:
        raise NotImplementedError

    def load_decoder(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        weights_dict: Mapping[str, Array],
    ) -> Decoder:
        config = self.to_decoder_config(context_length, activation_precision, accumulation_precision)
        model = config.empty()
        return self._load_decoder_weights(model, weights_dict)


@dataclass(frozen=True)
class ForeignClassifierConfig(ForeignConfig, RegistryABC):
    def to_classifier_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> ClassifierConfig:
        raise NotImplementedError

    @classmethod
    def _load_classifier_weights(
        cls,
        model: Classifier,
        weights_dict: Mapping[str, Array],
    ) -> Classifier:
        raise NotImplementedError

    def load_classifier(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        weights_dict: Mapping[str, Array],
    ) -> Classifier:
        config = self.to_classifier_config(context_length, activation_precision, accumulation_precision)
        model = config.empty(skip_pre_attention_norm=True)
        return self._load_classifier_weights(model, weights_dict)
