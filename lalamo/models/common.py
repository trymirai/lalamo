from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import equinox as eqx
from jax import numpy as jnp

from lalamo.common import DTypeLike
from lalamo.message_processor import Message, MessageProcessor, MessageProcessorConfig, UserMessage
from lalamo.modules import Classifier, Decoder, LalamoModule
from lalamo.modules.classifier import ClassifierConfig, ClassifierResult
from lalamo.modules.decoder import DecoderConfig, DecoderResult

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
    ) -> "TextModel": ...


class TextModel[ModelT: Decoder | Classifier](LalamoModule):
    model: ModelT
    message_processor: MessageProcessor = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.model.activation_precision

    def record_trace(self, messages: Iterable[Message] | None = None) -> ClassifierResult | DecoderResult:
        if messages is None:
            messages = [UserMessage("Tell me about London")]

        token_ids = jnp.array(self.message_processor.tokenize_request(messages))[None, :]
        _, num_tokens = token_ids.shape
        token_positions = jnp.arange(num_tokens)[None, :]
        return self.model(token_ids=token_ids, token_positions=token_positions, return_activation_trace=True)
