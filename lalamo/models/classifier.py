from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

from lalamo.message_processor import Message, MessageProcessor
from lalamo.modules import Classifier, ClassifierConfig, LalamoModule

from .common import TextModel, TextModelConfig

__all__ = [
    "ClassifierModel",
    "ClassifierModelConfig",
]


@dataclass(frozen=True)
class ClassifierModelConfig(TextModelConfig[ClassifierConfig]):
    def init(
        self,
        model: LalamoModule,
        message_processor: MessageProcessor,
    ) -> "ClassifierModel":
        assert isinstance(model, Classifier)
        return ClassifierModel(self, model, message_processor)

    @classmethod
    def load_model(cls, path: Path | str) -> "ClassifierModel":
        result = super().load_model(path)
        assert isinstance(result, ClassifierModel)
        return result


class ClassifierModel(TextModel[ClassifierModelConfig, Classifier]):
    def label_output_logits(self, logits: Float[Array, "batch logits"]) -> dict[str, Float[Array, " batch"]]:
        output_labels = self.model.config.output_labels
        probabilities = jax.nn.sigmoid(logits)

        if output_labels is None:
            output_labels = [f"class_{idx}" for idx in range(self.model.config.num_labels)]

        assert probabilities.ndim == 2, f"Expected 2D array, got array of shape {logits.shape}"

        return dict(zip(output_labels, jnp.unstack(probabilities, axis=1), strict=True))

    def classify_chat(
        self,
        messages: Iterable[Message],
    ) -> dict[str, float]:
        token_ids = jnp.array(self.message_processor.tokenize_request(messages), dtype=jnp.int32)[None, :]
        _, sequence_length = token_ids.shape
        token_positions = jnp.arange(sequence_length, dtype=jnp.int32)[None, :]
        classifier_output = self.model(token_ids=token_ids, token_positions=token_positions)

        return {k: float(v.item()) for k, v in self.label_output_logits(classifier_output.logits).items()}
