from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Self, TypedDict

import jax
from jaxtyping import Array, DTypeLike, PRNGKeyArray
from jinja2 import Template
from tokenizers import Tokenizer

from lalamo.common import ParameterTree
from lalamo.modules.common import LalamoModule

from .audio_decoder import TTSAudioDecoder
from .text_decoder import TTSTextDecoder
from .vocoders import Vocoder

__all__ = ["TTSMessage", "TTSRequestFactory", "TTSRequestFactoryConfig"]


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

    @staticmethod
    def simple_message(text: str) -> "TTSMessage":
        return TTSMessage(content=text, speaker_id="speaker:1", style="interleave")


class TTSRequest(TypedDict):
    messages: list[TTSMessage]


@dataclass(frozen=True)
class TTSRequestFactoryConfig:
    prompt_template: str

    # TODO(peter.glushkov): find a better way to handle opening new-line symbol
    drop_initial_newline: bool = True

    def init(self, tokenizer: Tokenizer) -> "TTSRequestFactory":
        return TTSRequestFactory(
            self,
            tokenizer=tokenizer,
        )


@dataclass(frozen=True)
class TTSRequestFactory:
    config: TTSRequestFactoryConfig
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


class ForeignTTSModelType(Enum):
    FISH_AUDIO = "fishaudio"
    FISH_AUDIO_LALAMO = "fishaudio_lalamo"


@dataclass(frozen=True)
class TTSConfig:
    text_decoder_config: Any
    audio_decoder_config: Any
    vocoder_config: Any

    activation_precision: DTypeLike

    def empty(self) -> "TTSModel":
        text_decoder = self.text_decoder_config.empty()
        audio_decoder = self.audio_decoder_config.empty()
        vocoder = self.vocoder_config.empty()
        return TTSModel(config=self, text_decoder=text_decoder, audio_decoder=audio_decoder, vocoder=vocoder)

    def random_init(self, key: PRNGKeyArray) -> "TTSModel":
        key1, key2 = jax.random.split(key)
        text_decoder = self.text_decoder_config.random_init(key1)
        audio_decoder = self.audio_decoder_config.random_init(key2)
        vocoder = self.vocoder_config.empty()
        return TTSModel(config=self, text_decoder=text_decoder, audio_decoder=audio_decoder, vocoder=vocoder)


class TTSModel(LalamoModule[TTSConfig]):
    text_decoder: TTSTextDecoder
    audio_decoder: TTSAudioDecoder
    vocoder: Vocoder

    @property
    def activation_precision(self) -> DTypeLike:
        return TTSConfig.activation_precision

    def export_weights(self) -> ParameterTree[Array]:
        return {}

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        return self
