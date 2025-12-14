from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self

import equinox as eqx
import numpy as np
import torch
from jax import Array
from jax import numpy as jnp
from jaxtyping import DTypeLike, Float

from lalamo.audio import AudioEncoding, AudioRenderer, AudioRenderingConfig
from lalamo.modules import AudioDecoder, LalamoModule, NoopVocoder, Vocoder, VocoderConfig
from lalamo.modules.audio.foreign.fish_audio import load_tokenizer_from_fish_audio
from lalamo.modules.audio.foreign.fish_audio_thin_wrapper import (
    FishAudioTextDecoderConfig_Foreign,
    load_fish_audio_audio_decoder,
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


def _build_foreign_fish_audio_model(
    path_to_checkpoints: Path, device: str = "cpu", precision: torch.dtype = torch.bfloat16
) -> "TTSGenerator":
    text_decoder_config = FishAudioTextDecoderConfig_Foreign.from_config_file(path_to_checkpoints / "config.json")
    text_decoder = text_decoder_config.load_model(path_to_checkpoints, device=device, precision=precision)
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

    return TTSGenerator(
        config=tts_config,
        text_decoder=text_decoder,
        audio_decoder=audio_decoder,
        vocoder=NoopVocoder(tts_config.vocoder_config),
        message_processor=message_processor,
        audio_renderer=AudioRenderer(audio_renderer_config),
    )


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

    def tokenize_text(self, messages: Iterable[TTSMessage]) -> Float[Array, " batch tokens"]:
        text_tokens = self.message_processor.tokenize_request(messages)
        return jnp.asarray(text_tokens[1:])[None, :]  # TODO: find a better way to avoid initial new-line symbol

    def decode_text(self, text_tokens: Float[Array, " batch tokens"]) -> Float[Array, " batch tokens channels"]:
        batch_size, sequence_length = text_tokens.shape
        token_ids = jnp.tile(jnp.arange(sequence_length), (batch_size, 1))

        return self.text_decoder(text_tokens)  # type: ignore

    def decode_audio(
        self, semantic_tokens: Float[Array, " batch tokens channels"]
    ) -> Float[Array, " batch audio_frames channels"]:
        # optionally reset the decoder
        return jnp.zeros((1, 2, 3))

    def generate_waveform(
        self, audio_features: Float[Array, " batch audio_frames channels"]
    ) -> Float[Array, " batch audio_channels audio_samples"]:
        return jnp.zeros(shape=(1, 2, 3))

    def generate_speech(self, messages: Iterable[TTSMessage]) -> np.ndarray:
        # 1. Process messages with .message_processor to get text tokens
        # 2. Encode text tokens into semantic tokens using .text_decoder
        # 3. Generate audio features from text tokens using .audio_decoder
        # 4. Generate final waveform from audio features using .vocoder
        # 5. Condition final audio waveform as required with .audio_renderer
        return np.zeros(5000)
