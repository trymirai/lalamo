from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Self

from jax import Array
from jax import numpy as jnp

from lalamo.common import DTypeLike, ParameterTree
from lalamo.message_processor import Message, MessageProcessor, MessageProcessorConfig
from lalamo.modules import Classifier, ClassifierConfig, LalamoModule


@dataclass
class RouterResult:
    message_labels: Mapping[str, float]


@dataclass(frozen=True)
class RouterConfig:
    classifier_config: ClassifierConfig
    message_processor_config: MessageProcessorConfig | None = None


class RouterModel(LalamoModule[RouterConfig]):
    classifier: Classifier
    message_processor: MessageProcessor | None = None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.classifier.activation_precision

    def export_weights(self) -> ParameterTree:
        return self.classifier.export_weights()

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        return replace(
            self,
            classifier=self.classifier.import_weights(weights),
        )

    def classify(
        self,
        messages: Iterable[Message],
    ) -> RouterResult:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(
            self.message_processor.tokenize(formatted_messages), dtype=jnp.int32
        )[None, :]
        batch_size, sequence_length = token_ids.shape
        token_positions = jnp.array([sequence_length - 1] * batch_size, dtype=jnp.int32)
        classifier_output = self.classifier(
            token_ids=token_ids, token_positions=token_positions
        )

        assert classifier_output.labels is not None
        return RouterResult(message_labels=classifier_output.labels)
