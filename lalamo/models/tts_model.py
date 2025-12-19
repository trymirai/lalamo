from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self, Optional

import equinox as eqx
import numpy as np
import torch
from jax import Array
from jax import numpy as jnp
from jaxtyping import DTypeLike, Float, Int

from lalamo.audio import AudioEncoding, AudioRenderer, AudioRenderingConfig, audio_rendering
from lalamo.modules import AudioDecoder, LalamoModule, NoopVocoder, Vocoder, VocoderConfig
from lalamo.modules.audio.foreign.fish_audio import FishAudioSamplingParams, load_tokenizer_from_fish_audio
from lalamo.modules.audio.foreign.fish_audio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
    FishAudioTextDecoderConfig_Foreign,
    load_fish_audio_audio_decoder,
    try_locate_fish_audio_model_path,
)
from lalamo.modules.audio.text_decoder import TextDecoder
from lalamo.modules.audio.tts_request_factory import TTSMessage, TTSRequestFactory, TTSRequestFactoryConfig

from .common import ParameterTree

__all__ = [
    "ForeignTTSModel",
    "TTSConfig",
    "TTSGenerator",
]


class ForeignTTSModel(Enum):
    FISH_AUDIO = "fishaudio"


@dataclass(frozen=True)
class TTSConfig:
    text_decoder_config: Any
    audio_decoder_config: Any
    vocoder_config: Any

    @classmethod
    def load_model_from_foreign_model_preset(
        cls, preset: ForeignTTSModel, path_to_checkpoints: Path
    ) -> "TTSGenerator":
        match preset:
            case ForeignTTSModel.FISH_AUDIO:
                return _build_foreign_fish_audio_model(path_to_checkpoints)

    @classmethod
    def try_locate_audio_model_path(cls, preset: ForeignTTSModel) -> Optional[Path]:
        match preset:
            case ForeignTTSModel.FISH_AUDIO:
                return try_locate_fish_audio_model_path()


@dataclass
class TTSGenerationResult:
    audio: np.ndarray
    audio_params: AudioRenderingConfig


class TTSGenerator(LalamoModule[TTSConfig]):
    config: TTSConfig

    message_processor: TTSRequestFactory = eqx.field(static=True)

    text_decoder: TextDecoder
    audio_decoder: AudioDecoder
    vocoder: Vocoder

    audio_renderer: AudioRenderer = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.text_decoder.activation_precision

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "text_encoder": self.text_decoder.export_weights(),
            "audio_decoder": self.text_decoder.export_weights(),
            "vocoder": self.text_decoder.export_weights(),
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        return self

    def tokenize_text(self, messages: Iterable[TTSMessage]) -> Int[Array, " batch tokens"]:
        text_tokens = self.message_processor.tokenize_request(messages)
        return jnp.asarray(text_tokens)[None, :]

    @abstractmethod
    def decode_text(self, text_tokens: Array) -> Array: ...

    @abstractmethod
    def decode_audio(self, semantic_tokens: Array) -> Array: ...

    @abstractmethod
    def generate_waveform(self, audio_features: Array) -> Array: ...

    @abstractmethod
    def get_generated_audio_params(self) -> AudioRenderingConfig: ...

    def generate_speech(self, messages: Iterable[TTSMessage]) -> TTSGenerationResult:
        text_tokens = self.tokenize_text(messages)

        semantic_tokens = self.decode_text(text_tokens)

        audio_features = self.audio_decoder(semantic_tokens)

        audio_waveform = self.vocoder(audio_features)

        audio_waveform = self.audio_renderer.condition_signal(
            generated_audio=np.array(audio_waveform),
            generated_audio_properties=self.get_generated_audio_params(),
        )

        return TTSGenerationResult(audio=audio_waveform, audio_params=self.audio_renderer.config)


class FishAudioTTSGenerator_Foreign(TTSGenerator):
    def decode_text(self, text_tokens: Array) -> Array:
        assert isinstance(self.text_decoder, FishAudioTextDecoder_Foreign)

        # TODO (peter.glushkov): think how to handle it better, either get them from
        # config or make it part of 'decode_text' interface somehow
        sampling_params = FishAudioSamplingParams(
            argmax_decoding=False, temperature=0.8008, top_p=0.8008, repetition_penalty=1.1016
        )

        return self.text_decoder.decode_utterance(text_tokens, sampling_params=sampling_params)

    def decode_audio(self, semantic_tokens: Array) -> Array:
        return self.audio_decoder(semantic_tokens)

    def generate_waveform(self, audio_features: Array) -> Array:
        return self.vocoder(audio_features)

    def get_generated_audio_params(self) -> AudioRenderingConfig:
        # NOTE: mb this could be moved to config level
        return AudioRenderingConfig(
            samplerate=self.audio_decoder.dac_model.sample_rate,
            output_channels=1,
            bitwidth=16,
            encoding=AudioEncoding.pcm,
        )


def _build_foreign_fish_audio_model(
    path_to_checkpoints: Path, device: str = "cpu", precision: torch.dtype = torch.bfloat16
) -> "TTSGenerator":
    text_decoder_config = FishAudioTextDecoderConfig_Foreign.from_config_file(path_to_checkpoints / "config.json")
    text_decoder: FishAudioTextDecoder_Foreign = text_decoder_config.load_model(
        path_to_checkpoints, device=device, precision=precision
    )
    audio_decoder = load_fish_audio_audio_decoder(path_to_checkpoints)

    tokenizer = load_tokenizer_from_fish_audio(str(path_to_checkpoints))

    prompt_template = """
{% for message in messages %}<|{{message.style}}|><|{{message.speaker_id}}|>{{message.content}}{% endfor %}
"""

    tts_request_factory_config = TTSRequestFactoryConfig(
        prompt_template=prompt_template,
    )

    message_processor = TTSRequestFactory(tts_request_factory_config, tokenizer)

    tts_config = TTSConfig(
        text_decoder.config,
        audio_decoder.config,
        VocoderConfig(),
    )

    audio_renderer_config = AudioRenderingConfig(44100, 1, 16, AudioEncoding.pcm)

    return FishAudioTTSGenerator_Foreign(
        config=tts_config,
        text_decoder=text_decoder,
        audio_decoder=audio_decoder,
        vocoder=NoopVocoder(tts_config.vocoder_config),
        message_processor=message_processor,
        audio_renderer=AudioRenderer(audio_renderer_config),
    )
