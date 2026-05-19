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
class TokenCodec[RequestT, ResponseT, ConfigT: TokenCodecConfig](RegistryABC):
    config: ConfigT
    tokenizer: Tokenizer

    @abstractmethod
    def encode_request(self, request: RequestT) -> list[int]: ...

    @abstractmethod
    def decode_response(self, response: list[int]) -> ResponseT: ...
