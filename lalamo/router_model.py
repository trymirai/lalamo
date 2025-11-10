import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

from jax import Array
from jax import numpy as jnp
from jaxtyping import Float
from tokenizers import Tokenizer

from lalamo.common import DTypeLike, ParameterTree, unflatten_parameters
from lalamo.message_processor import Message, MessageProcessor, MessageProcessorConfig
from lalamo.modules import Classifier, ClassifierConfig, LalamoModule, config_converter
from lalamo.utils import open_safetensors


@dataclass
class RouterResult:
    message_labels: Mapping[str, Float]


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

    @classmethod
    def load(cls, path: Path | str) -> Self:
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)
        config = config_converter.structure(config_json["model_config"], RouterConfig)
        with open_safetensors(path / "model.safetensors") as weights_dict:
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
        token_ids = jnp.array(
            self.message_processor.tokenize(formatted_messages), dtype=jnp.int32
        )[None, :]
        batch_size, sequence_length = token_ids.shape
        token_positions = jnp.tile(
            jnp.arange(sequence_length, dtype=jnp.int32), (batch_size, 1)
        )
        classifier_output = self.classifier(
            token_ids=token_ids, token_positions=token_positions
        )

        if len(classifier_output.labels) > 1:
            raise ValueError("Expecting to have only single output batch")
        return RouterResult(message_labels=classifier_output.labels[0])
