from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import Int  # noqa: TC002
from tokenizers import Tokenizer

from lalamo.audio.audio_rendering import AudioEncoding, AudioRenderingSettings
from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.tts_codec import TTSCodec, TTSCodecConfig, TTSMessage
from lalamo.module import Keychain, LalamoConfig
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfig
from lalamo.modules.audio.text_decoder import TTSTextDecoder, TTSTextDecoderConfig
from lalamo.modules.audio.vocoders import Vocoder, VocoderConfig
from lalamo.modules.decoder import DecoderForwardPassConfig
from lalamo.sampling import SamplingPolicy

__all__ = [
    "TTSConfig",
    "TTSGenerationResult",
    "TTSMessage",
    "TTSModel",
    "TTSModelConfig",
]


@dataclass(frozen=True)
class TTSConfig(LalamoConfig):
    text_decoder_config: TTSTextDecoderConfig
    audio_decoder_config: TTSAudioDecoderConfig
    vocoder_config: VocoderConfig


@dataclass(frozen=True)
class TTSModelConfig(ModelConfig[TTSCodecConfig]):
    tts_config: TTSConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "TTSModel":
        token_codec = self.token_codec_config.init(tokenizer)
        return TTSModel(
            config=self,
            token_codec=token_codec,
            text_decoder=self.tts_config.text_decoder_config.init(initializer),
            audio_decoder=self.tts_config.audio_decoder_config.init(initializer),
            vocoder=self.tts_config.vocoder_config.init(initializer),
        )


@dataclass(frozen=True)
class TTSGenerationResult:
    audio: np.ndarray
    audio_params: AudioRenderingSettings


class TTSModel(Model[TTSCodecConfig, TTSModelConfig, TTSCodec]):
    token_codec: TTSCodec
    text_decoder: TTSTextDecoder
    audio_decoder: TTSAudioDecoder
    vocoder: Vocoder

    def get_generated_audio_params(self) -> AudioRenderingSettings:
        return AudioRenderingSettings(
            samplerate=self.audio_decoder.samplerate,
            output_channels=1,
            bitwidth=16,
            encoding=AudioEncoding.PCM,
        )

    def decode_utterance(
        self,
        text_tokens: Array,
        sampling_policy: SamplingPolicy | None = None,
        *,
        keychain: Keychain,
        forward_pass_config: DecoderForwardPassConfig = DecoderForwardPassConfig(),
    ) -> Array:
        return self.text_decoder.decode_utterance(
            text_tokens,
            sampling_policy=sampling_policy,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
        )

    def generate_speech(
        self,
        messages: Iterable[TTSMessage],
        sampling_policy: SamplingPolicy | None = None,
        *,
        keychain: Keychain,
        forward_pass_config: DecoderForwardPassConfig = DecoderForwardPassConfig(),
    ) -> TTSGenerationResult:
        text_keychain, audio_keychain = keychain.split()
        text_tokens: Int[Array, "batch tokens"] = jnp.asarray(
            self.token_codec.encode_request(messages),
            dtype=jnp.int32,
        )[None, :]
        semantic_tokens = self.decode_utterance(
            text_tokens,
            sampling_policy=sampling_policy,
            forward_pass_config=forward_pass_config,
            keychain=text_keychain,
        )
        audio_features = self.audio_decoder.audio_from_codes(semantic_tokens, keychain=audio_keychain)
        audio_waveform = self.vocoder(audio_features)

        return TTSGenerationResult(
            audio=np.asarray(audio_waveform),
            audio_params=self.get_generated_audio_params(),
        )
