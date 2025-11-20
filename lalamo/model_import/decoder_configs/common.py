from abc import abstractmethod
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.modules import Classifier, ClassifierConfig, Decoder, DecoderConfig
from lalamo.registry_abc import RegistryABC
from lalamo.modules.common import LalamoModule

__all__ = ["ForeignClassifierConfig", "ForeignLMConfig"]


@dataclass(frozen=True)
class ForeignConfig[ConfigT: DecoderConfig | ClassifierConfig](RegistryABC):
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    @property
    def eos_token_ids(self) -> list[int]:
        raise NotImplementedError

    @property
    def default_precision(self) -> DTypeLike:
        return jnp.dtype(getattr(self, "torch_dtype", "bfloat16"))

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
    ) -> LalamoModule:
        raise NotImplementedError

    @abstractmethod
    def to_lalamo_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> ConfigT:
        raise NotImplementedError

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
    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig:
        raise NotImplementedError

    @property
    def eos_token_ids(self) -> list[int]:
        return [self.eos_token_id] if isinstance(self.eos_token_id, int) else self.eos_token_id

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
    def to_classifier_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> ClassifierConfig:
        raise NotImplementedError

    def to_lalamo_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> ClassifierConfig:
        return self.to_classifier_config(context_length, activation_precision, accumulation_precision)
