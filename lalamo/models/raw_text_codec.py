from dataclasses import dataclass

from tokenizers import Tokenizer

from lalamo.token_codec import TokenCodec, TokenCodecConfig

__all__ = [
    "RawTextCodec",
    "RawTextCodecConfig",
]


@dataclass(frozen=True)
class RawTextCodecConfig(TokenCodecConfig):
    def init(self, tokenizer: Tokenizer) -> "RawTextCodec":
        return RawTextCodec(config=self, tokenizer=tokenizer)


@dataclass(frozen=True)
class RawTextCodec(TokenCodec[str, str, RawTextCodecConfig]):
    def encode_request(self, request: str) -> list[int]:
        return self.tokenizer.encode(request, add_special_tokens=False).ids

    def decode_response(self, response: list[int]) -> str:
        return self.tokenizer.decode(response, skip_special_tokens=False)
