from dataclasses import dataclass

from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.models.language_model import GenerationConfig
from lalamo.module import LalamoConfig
from lalamo.modules.decoder import DecoderConfig

__all__ = [
    "DecisionConfig",
    "DecisionModel",
    "DecisionModelConfig",
]


@dataclass(frozen=True)
class DecisionConfig(LalamoConfig):
    codes: tuple[str, ...]
    temperature: float


@dataclass(frozen=True)
class DecisionModelConfig(ModelConfig[ChatCodecConfig]):
    decoder_config: DecoderConfig
    generation_config: GenerationConfig
    decision: DecisionConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "DecisionModel":
        raise NotImplementedError("Decision model inference is not supported in Lalamo.")


class DecisionModel(Model[DecisionModelConfig, ChatCodec]):
    pass
