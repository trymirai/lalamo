from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

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
class TTSCodec(TokenCodec[TTSCodecConfig]):
    @cached_property
    def prompt_template(self) -> Template:
        return Template(self.config.prompt_template)

    def render_request(self, messages: Iterable[TTSMessage]) -> str:
        prompt_text = self.prompt_template.render(messages=list(messages))
        if self.config.drop_initial_newline and prompt_text.startswith("\n"):
            prompt_text = prompt_text[1:]
        return prompt_text
