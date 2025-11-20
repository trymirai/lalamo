import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float
from tokenizers import Tokenizer

from lalamo.common import DTypeLike, ParameterTree, unflatten_parameters
from lalamo.message_processor import Message, MessageProcessor, MessageProcessorConfig, tokenize_message
from lalamo.modules import Classifier, ClassifierConfig, LalamoModule, config_converter
from lalamo.modules.classifier import ClassifierResult
from lalamo.utils import open_safetensors

from .utils import get_dummy_tokens


@dataclass
class RouterResult:
    message_labels: list[dict[str, float]]


@dataclass(frozen=True)
class RouterConfig:
    classifier_config: ClassifierConfig
    message_processor_config: MessageProcessorConfig


class RouterModel(LalamoModule[RouterConfig]):
    classifier: Classifier
    message_processor: MessageProcessor

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

    def label_output_logits(self, jax_logits: Float[Array, " logits"]) -> dict[str, float]:
        config = self.classifier.config
        sigmoids = np.asarray(jax.nn.sigmoid(jax_logits))

        labels = (
            config.output_labels
            if config.output_labels is not None
            else [f"class_{idx}" for idx in range(config.num_labels)]
        )

        n_labels = len(labels)
        if n_labels != sigmoids.shape[0]:
            raise ValueError("Number of output logits is different from provided list of labels")

        return {labels[idx]: sigmoids[idx] for idx in range(n_labels)}

    @classmethod
    def load(cls, path: Path | str) -> Self:
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)
        config = config_converter.structure(config_json["model_config"], RouterConfig)
        with open_safetensors(path / "model.safetensors") as open_results:
            weights_dict, _ = open_results
            weights = unflatten_parameters(weights_dict)
            decoder = config.classifier_config.empty().import_weights(weights)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = MessageProcessor(config.message_processor_config, tokenizer)
        return cls(config, decoder, message_processor)

    def classify(
        self,
        message: Message,
    ) -> RouterResult:
        formatted_messages = self.message_processor.render_request(messages=[message])
        token_ids = jnp.array(self.message_processor.tokenize(formatted_messages), dtype=jnp.int32)[None, :]
        batch_size, sequence_length = token_ids.shape
        token_positions = jnp.tile(jnp.arange(sequence_length, dtype=jnp.int32), (batch_size, 1))
        classifier_output = self.classifier(token_ids=token_ids, token_positions=token_positions)

        labels = [self.label_output_logits(classifier_output.logits[batch]) for batch in range(batch_size)]

        return RouterResult(message_labels=labels)

    def record_trace(self, message: Message | None = None) -> ClassifierResult:
        if message:
            token_ids, token_positions = tokenize_message(self.message_processor, message)
        else:
            token_ids, token_positions = get_dummy_tokens()
        return self.classifier(token_ids=token_ids, token_positions=token_positions)
