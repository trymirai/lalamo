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
from lalamo.module import Keychain
from lalamo.modules.audio.text_to_speech import TTSConfig, TTSModel
from lalamo.sampling import SamplingPolicy

__all__ = [
    "TTSGenerationResult",
    "TTSGenerator",
    "TTSGeneratorConfig",
    "TTSMessage",
]


@dataclass(frozen=True)
class TTSGeneratorConfig(ModelConfig[TTSCodecConfig]):
    tts_config: TTSConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "TTSGenerator":
        tts_model = self.tts_config.init(initializer)
        token_codec = self.token_codec_config.init(tokenizer)
        return TTSGenerator(self, token_codec, tts_model)


@dataclass(frozen=True)
class TTSGenerationResult:
    audio: np.ndarray
    audio_params: AudioRenderingSettings


class TTSGenerator(Model[TTSCodecConfig, TTSGeneratorConfig, TTSCodec]):
    token_codec: TTSCodec
    tts_model: TTSModel

    def get_generated_audio_params(self) -> AudioRenderingSettings:
        return AudioRenderingSettings(
            samplerate=self.tts_model.audio_decoder.samplerate,
            output_channels=1,
            bitwidth=16,
            encoding=AudioEncoding.PCM,
        )

    def generate_speech(
        self,
        messages: Iterable[TTSMessage],
        sampling_policy: SamplingPolicy | None = None,
        *,
        keychain: Keychain,
    ) -> TTSGenerationResult:
        text_keychain, audio_keychain = keychain.split()
        text_tokens: Int[Array, "batch tokens"] = jnp.asarray(
            self.token_codec.encode_request(messages),
            dtype=jnp.int32,
        )[None, :]
        semantic_tokens = self.tts_model.text_decoder.decode_utterance(
            text_tokens,
            sampling_policy=sampling_policy,
            keychain=text_keychain,
        )
        audio_features = self.tts_model.audio_decoder.audio_from_codes(semantic_tokens, keychain=audio_keychain)
        audio_waveform = self.tts_model.vocoder(audio_features)

        return TTSGenerationResult(
            audio=np.asarray(audio_waveform),
            audio_params=self.get_generated_audio_params(),
        )
