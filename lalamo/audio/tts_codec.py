from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import TypedDict

from jinja2 import Template
from tokenizers import Tokenizer

from lalamo.token_codec import TokenCodec, TokenCodecConfig

__all__ = [
    "TTSCodec",
    "TTSCodecConfig",
    "TTSMessage",
]


@dataclass(frozen=True)
class TTSMessage:
    content: str
    speaker_id: str
    style: str


class TTSRequest(TypedDict):
    messages: list[TTSMessage]


@dataclass(frozen=True)
class TTSCodecConfig(TokenCodecConfig):
    prompt_template: str
    drop_initial_newline: bool = True

    def init(self, tokenizer: Tokenizer) -> "TTSCodec":
        return TTSCodec(
            config=self,
            tokenizer=tokenizer,
        )


@dataclass(frozen=True)
class TTSCodec(TokenCodec[Iterable[TTSMessage], str, TTSCodecConfig]):
    @cached_property
    def prompt_template(self) -> Template:
        return Template(self.config.prompt_template)

    def request_to_dict(
        self,
        messages: Iterable[TTSMessage],
    ) -> TTSRequest:
        return TTSRequest(messages=list(messages))

    def render_request(self, messages: Iterable[TTSMessage]) -> str:
        request_dict = self.request_to_dict(messages)
        prompt_text = self.prompt_template.render({**request_dict})
        if self.config.drop_initial_newline and prompt_text.startswith("\n"):
            prompt_text = prompt_text[1:]
        return prompt_text

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_request(self, request: Iterable[TTSMessage]) -> list[int]:
        return self.encode_text(self.render_request(request))

    def decode_response(self, response: list[int]) -> str:
        return self.tokenizer.decode(response, skip_special_tokens=False)
