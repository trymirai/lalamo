from collections.abc import Iterable
from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float
from tokenizers import Tokenizer

from lalamo.chat_codec import ChatCodec, ChatCodecConfig, Message
from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.modules import Classifier, ClassifierConfig, Keychain

__all__ = [
    "ClassifierModel",
    "ClassifierModelConfig",
]


@dataclass(frozen=True)
class ClassifierModelConfig(ModelConfig[ChatCodecConfig]):
    classifier_config: ClassifierConfig
    output_labels: tuple[str, ...] | None = None

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "ClassifierModel":
        classifier = self.classifier_config.init(initializer)
        token_codec = self.token_codec_config.init(tokenizer)
        return ClassifierModel(self, token_codec, classifier)


class ClassifierModel(Model[ChatCodecConfig, ClassifierModelConfig, ChatCodec]):
    token_codec: ChatCodec
    classifier: Classifier

    def label_output_logits(self, logits: Float[Array, "batch logits"]) -> dict[str, Float[Array, " batch"]]:
        output_labels = self.config.output_labels or self.classifier.config.output_labels
        probabilities = jax.nn.sigmoid(logits)

        if output_labels is None:
            output_labels = [f"class_{idx}" for idx in range(self.classifier.config.num_labels)]

        assert probabilities.ndim == 2, f"Expected 2D array, got array of shape {logits.shape}"

        return dict(zip(output_labels, jnp.unstack(probabilities, axis=1), strict=True))

    def classify_chat(
        self,
        messages: Iterable[Message],
        *,
        keychain: Keychain,
    ) -> dict[str, float]:
        token_ids = jnp.array(self.token_codec.encode_request(messages), dtype=jnp.int32)[None, :]
        _, sequence_length = token_ids.shape
        token_positions = jnp.arange(sequence_length, dtype=jnp.int32)[None, :]
        classifier_output = self.classifier(
            token_ids=token_ids,
            token_positions=token_positions,
            keychain=keychain,
        )

        return {k: float(v.item()) for k, v in self.label_output_logits(classifier_output.logits).items()}
