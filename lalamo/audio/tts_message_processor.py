from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import TypedDict

from jinja2 import Template
from tokenizers import Tokenizer


@dataclass(frozen=True)
class VoicePrompt:
    """
    Current class is reserved for future usage of audio prompts
    to condition style of generated audio
    """


@dataclass(frozen=True)
class TTSMessage:
    content: str
    speaker_id: str
    style: str


class TTSRequest(TypedDict):
    messages: list[TTSMessage]


@dataclass(frozen=True)
class TTSMessageProcessorConfig:
    prompt_template: str

    # TODO(peter.glushkov): find a better way to handle opening new-line symbol
    drop_initial_newline: bool = True

    def init(self, tokenizer: Tokenizer) -> "TTSMessageProcessor":
        return TTSMessageProcessor(
            self,
            tokenizer=tokenizer,
        )


@dataclass(frozen=True)
class TTSMessageProcessor:
    config: TTSMessageProcessorConfig
    tokenizer: Tokenizer

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

    def tokenize_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def tokenize_request(self, messages: Iterable[TTSMessage]) -> list[int]:
        rendered = self.render_request(messages)
        return self.tokenize_text(rendered)

    def detokenize(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)
