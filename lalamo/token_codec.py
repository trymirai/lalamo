from abc import abstractmethod
from dataclasses import dataclass

from tokenizers import Tokenizer

from lalamo.module import LalamoConfig
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "TokenCodec",
    "TokenCodecConfig",
]


@dataclass(frozen=True)
class TokenCodecConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, tokenizer: Tokenizer) -> "TokenCodec": ...


@dataclass(frozen=True)
class TokenCodec[ConfigT: TokenCodecConfig]:
    config: ConfigT
    tokenizer: Tokenizer

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids
