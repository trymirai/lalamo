from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import jax
import equinox as eqx
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, DTypeLike

from lalamo.message_processor import Message, MessageProcessor
from lalamo.modules import Decoder, LalamoModule, AudioDecoder, Vocoder
from lalamo.audio import AudioRenderer

from .common import TextModel, TextModelConfig, ParameterTree, Message

__all__ = [
    "TTSGenerator",
    "TTSConfig",
]


@dataclass(frozen=True)
class TTSConfig:

    @classmethod
    def load_model(cls, path: Path | str) -> "TTSGenerator":
        text_encoder_config = TextEncoderConfig

        return TTSGenerator(
        )


class TTSGenerator(LalamoModule[TTSConfig]):
    config: TTSConfig
    text_encoder: Decoder
    audio_decoder: AudioDecoder
    vocoder: Vocoder

    message_processor: MessageProcessor = eqx.field(static=True)
    audio_rendered: AudioRenderer = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.text_encoder.activation_precision

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "text_encoder": self.text_encoder.export_weights(),
            "audio_decoder": self.text_encoder.export_weights(),
            "vocoder": self.text_encoder.export_weights(),
        }

    def _tokenize_text(self, messages: Iterable[Message]) -> Float[Array, " batch tokens"]:
        return Array(self.message_processor.tokenize_request(messages))[None, :]

    def encode_text(self, text_tokens: Float[Array, " batch tokens"]) -> Float[Array, " batch tokens channels"]:
        batch_size, sequence_length = text_tokens.shape
        token_ids = jnp.tile(jnp.arange(sequence_length), (batch_size, 1))

        return self.text_encoder(text_tokens, token_ids)  # type: ignore

    def decode_audio(
        self, semantic_tokens: Float[Array, " batch tokens channels"]
    ) -> Float[Array, " batch audio_frames channels"]:
        # optionally reset the decoder


    def generate_waveform(
        self, audio_features: Float[Array, " batch audio_frames channels"]
    ) -> Float[Array, " batch audio_channels audio_samples"]:
        return jnp.zeros(shape=(1, 2, 3))

    def generate_speech(self, messages: Iterable[Message]) -> None:
        # 1. Process messages with .message_processor to get text tokens
        # 2. Encode text tokens into semantic tokens using .text_encoder
        # 3. Generate audio features from text tokens using .audio_decoder
        # 4. Generate final waveform from audio features using .vocoder
        # 5. Condition final audio waveform as required with .audio_renderer
        pass
