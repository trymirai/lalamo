import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

import equinox as eqx
import jax
from jax import numpy as jnp
from tokenizers import Tokenizer

from lalamo.common import ParameterPath, is_abstract_array
from lalamo.initializer import EmptyInitializer
from lalamo.message_processor import Message, MessageProcessor, MessageProcessorConfig, UserMessage
from lalamo.modules import Classifier, Decoder, Keychain, LalamoModule, config_converter
from lalamo.modules.classifier import ClassifierConfig, ClassifierResult
from lalamo.modules.decoder import DecoderConfig, DecoderResult
from lalamo.safetensors import safe_read

__all__ = [
    "BatchSizeInfo",
    "BatchSizesComputedEvent",
    "TextModel",
    "TextModelConfig",
]


@dataclass(frozen=True)
class InferenceConfig:
    max_output_length: int = 8192
    padded_length: int = 8192
    num_top_logits_to_return: int | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class BatchSizeInfo:
    prefix_length: int
    num_elements: int
    batch_size: int


@dataclass(frozen=True)
class BatchSizesComputedEvent:
    batch_sizes: tuple[BatchSizeInfo, ...]


@dataclass(frozen=True)
class TextModelConfig[ConfigT: ClassifierConfig | DecoderConfig](ABC):
    model_config: ConfigT
    message_processor_config: MessageProcessorConfig

    @abstractmethod
    def init(
        self,
        model: LalamoModule,
        message_processor: MessageProcessor,
    ) -> LalamoModule: ...

    @classmethod
    def load_model(cls, path: Path | str) -> LalamoModule:
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)
        config = config_converter.structure(config_json["model_config"], cls)
        with Path(path / "model.safetensors").open("rb") as fd:
            metadata, tensors = safe_read(fd)
            weights_dict = {**tensors, **(metadata or {})}
            if all(key.startswith("model.") for key in tensors):
                weights_dict = {key.removeprefix("model."): value for key, value in weights_dict.items()}
            model = config.model_config.init(EmptyInitializer(precision=jnp.float32)).from_uzu(weights_dict)  # type: ignore[attr-defined]
        abstract_leaves = [
            f"{ParameterPath('') / path}: shape={leaf.shape}, dtype={leaf.dtype}"
            for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]
            if is_abstract_array(leaf)
        ]
        if abstract_leaves:
            raise ValueError(
                "Model contains abstract leaves after loading:\n" + "\n".join(abstract_leaves),
            )
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = MessageProcessor(config.message_processor_config, tokenizer)
        return config.init(model, message_processor)


class TextModel[ConfigT, ModelT: Decoder | Classifier](LalamoModule[ConfigT]):
    model: ModelT
    message_processor: MessageProcessor = eqx.field(static=True)

    def record_trace(self, messages: Iterable[Message] | None = None) -> ClassifierResult | DecoderResult:
        if messages is None:
            messages = [UserMessage("Tell me about London")]

        token_ids = jnp.array(self.message_processor.tokenize_request(messages))[None, :]
        _, num_tokens = token_ids.shape
        token_positions = jnp.arange(num_tokens)[None, :]
        return self.model(
            token_ids=token_ids,
            token_positions=token_positions,
            return_activation_trace=True,
            keychain=Keychain.init(0),
        )
