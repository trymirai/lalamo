from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Self, TypedDict

import jax
from jaxtyping import Array, DTypeLike, PRNGKeyArray
from jinja2 import Template
from tokenizers import Tokenizer

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.common import LalamoModule, register_config_union
from lalamo.sampling import SamplingPolicy, make_policy

from .audio_decoder import TTSAudioDecoder
from .fishaudio.fishaudio_audio_decoding import DescriptAudioCodecConfig
from .fishaudio.fishaudio_text_decoding import FishAudioTextDecoderConfig
from .nanocodec.audio_decoding import NanoCodecConfig
from .nanocodec.stub_text_decoder import StubTextDecoderConfig
from .text_decoder import TTSTextDecoder
from .vocoders import Vocoder, VocoderConfig

__all__ = ["TTSMessage", "TTSMessageProcessor", "TTSMessageProcessorConfig"]

DEFAULT_TTS_SAMPLING_POLICY: SamplingPolicy = make_policy(temperature=0.3, top_p=0.9)
DEFAULT_TTS_REPETITION_PENALTY: float = 1.1


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


TTSAudioDecoderConfig = DescriptAudioCodecConfig | NanoCodecConfig
register_config_union(TTSAudioDecoderConfig)

TTSTextDecoderConfig = FishAudioTextDecoderConfig | StubTextDecoderConfig
register_config_union(TTSTextDecoderConfig)


@dataclass(frozen=True)
class TTSConfig:
    text_decoder_config: TTSTextDecoderConfig
    audio_decoder_config: TTSAudioDecoderConfig
    vocoder_config: VocoderConfig

    activation_precision: DTypeLike

    def empty(self) -> "TTSModel":
        text_decoder = self.text_decoder_config.empty()
        audio_decoder = self.audio_decoder_config.empty()
        vocoder = self.vocoder_config.empty()
        return TTSModel(config=self, text_decoder=text_decoder, audio_decoder=audio_decoder, vocoder=vocoder)

    def random_init(self, key: PRNGKeyArray) -> "TTSModel":
        key_text, key_audio = jax.random.split(key)
        text_decoder = self.text_decoder_config.random_init(key=key_text)
        audio_decoder = self.audio_decoder_config.random_init(key=key_audio)
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
        assert isinstance(weights, Mapping)
        return replace(
            self,
            text_decoder=self.text_decoder.import_weights(require_tree(weights["text_decoder"])),
            audio_decoder=self.audio_decoder.import_weights(require_tree(weights["audio_decoder"])),
        )
