import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

import equinox as eqx
from jax import Array
from jax import numpy as jnp
from tokenizers import Tokenizer

from lalamo.common import DTypeLike, ParameterTree, unflatten_parameters
from lalamo.message_processor import Message, MessageProcessor, MessageProcessorConfig, UserMessage
from lalamo.modules import Classifier, Decoder, LalamoModule, config_converter
from lalamo.modules.classifier import ClassifierConfig, ClassifierResult
from lalamo.modules.decoder import DecoderConfig, DecoderResult
from lalamo.safetensors import safe_read

__all__ = [
    "TextModel",
    "TextModelConfig",
]


@dataclass(frozen=True)
class TextModelConfig[ConfigT: ClassifierConfig | DecoderConfig](ABC):
    model_config: ConfigT
    message_processor_config: MessageProcessorConfig

    @abstractmethod
    def init(
        self,
        model: LalamoModule,
        message_processor: MessageProcessor,
    ) -> LalamoModule[Self]: ...

    @classmethod
    def load_model(cls, path: Path | str) -> LalamoModule[Self]:
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)
        config = config_converter.structure(config_json["model_config"], cls)
        with Path(path / "model.safetensors").open("rb") as fd:
            _, weights_dict = safe_read(fd)
            weights = unflatten_parameters(weights_dict)
            model = config.model_config.empty().import_weights(weights)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = MessageProcessor(config.message_processor_config, tokenizer)
        return config.init(model, message_processor)


class TextModel[ConfigT, ModelT: Decoder | Classifier](LalamoModule[ConfigT]):
    model: ModelT
    message_processor: MessageProcessor = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.model.activation_precision

    def export_weights(self) -> ParameterTree:
        return self.model.export_weights()

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        return replace(
            self,
            model=self.model.import_weights(weights),
        )

    def record_trace(self, messages: Iterable[Message] | None = None) -> ClassifierResult | DecoderResult:
        if messages is None:
            messages = [UserMessage("Tell me about London")]

        token_ids = jnp.array(self.message_processor.tokenize_request(messages))[None, :]
        _, num_tokens = token_ids.shape
        token_positions = jnp.arange(num_tokens)[None, :]
        return self.model(token_ids=token_ids, token_positions=token_positions, return_activation_trace=True)
