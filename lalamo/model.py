from abc import abstractmethod
from dataclasses import dataclass

from tokenizers import Tokenizer

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule, field
from lalamo.token_codec import TokenCodec, TokenCodecConfig
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "Model",
    "ModelConfig",
]


@dataclass(frozen=True)
class ModelConfig[TokenCodecConfigT: TokenCodecConfig](LalamoConfig, RegistryABC):
    token_codec_config: TokenCodecConfigT

    @abstractmethod
    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "Model": ...


class Model[
    TokenCodecConfigT: TokenCodecConfig,
    ConfigT: ModelConfig,
    TokenCodecT: TokenCodec,
](LalamoModule[ConfigT], Exportable):
    token_codec: TokenCodecT = field(static=True)
