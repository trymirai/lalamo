import json
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer

from lalamo.initializer import EmptyInitializer
from lalamo.model import Model, ModelConfig
from lalamo.models import (
    ClassifierModelConfig,
    GenerationConfig,
    LanguageModelConfig,
    TTSConfig,
    TTSModelConfig,
)
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.modules.classifier import ClassifierConfig
from lalamo.modules.decoder import DecoderConfig
from lalamo.utils.registry_abc import RegistryABC
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import CompressionImplementation

__all__ = ["ForeignClassifierConfig", "ForeignConfig", "ForeignLMConfig", "ForeignTTSConfig"]


@dataclass(frozen=True)
class ForeignConfig[ConfigT: ModelConfig](RegistryABC):
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)

    @property
    @abstractmethod
    def default_dtype(self) -> DTypeLike: ...

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)

    @abstractmethod
    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model: ...

    def load(
        self,
        config: ConfigT,
        tokenizer: Tokenizer,
        dtype: DTypeLike,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
    ) -> Model:
        model = config.init(
            tokenizer=tokenizer,
            initializer=EmptyInitializer(dtype=dtype, sharding_config=sharding_config),
        )
        return self._load_weights(model=model, weights_dict=weights_dict, implementation=implementation)


@dataclass(frozen=True)
class ForeignLMConfig(ForeignConfig[LanguageModelConfig], RegistryABC):
    @abstractmethod
    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig: ...

    @property
    @abstractmethod
    def eos_token_ids(self) -> list[int]: ...

    def to_lalamo_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],
        token_codec_config: ChatCodecConfig,
        generation_config: GenerationConfig,
    ) -> LanguageModelConfig:
        return LanguageModelConfig(
            token_codec_config=token_codec_config,
            decoder_config=self.to_decoder_config(context_length, metadata_dict),
            generation_config=generation_config,
        )


@dataclass(frozen=True)
class ForeignClassifierConfig(ForeignConfig[ClassifierModelConfig], RegistryABC):
    @abstractmethod
    def to_classifier_config(
        self,
        context_length: int | None,
    ) -> ClassifierConfig: ...


@dataclass(frozen=True)
class ForeignTTSConfig(ForeignConfig[TTSModelConfig], RegistryABC):
    @abstractmethod
    def to_tts_config(
        self,
        context_length: int | None,
    ) -> TTSConfig: ...
