import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import DTypeLike, Float, Int, PRNGKeyArray
from tokenizers import Tokenizer

from lalamo.audio.audio_rendering import AudioEncoding, AudioRenderingSettings
from lalamo.audio.utils import DEFAULT_SAMPLERATE
from lalamo.modules import TTSModel, config_converter
from lalamo.modules.audio.fishaudio.fishaudio_common import (
    DEFAULT_FISHAUDIO_RANDOM_SEED,
    FishaudioConsts,
    default_fishaudio_sampling_policy,
)
from lalamo.modules.audio.fishaudio.fishaudio_text_decoding import (
    FishAudioTextDecoder,
)
from lalamo.modules.audio.text_to_speech import (
    DEFAULT_TTS_REPETITION_PENALTY,
    DEFAULT_TTS_SAMPLING_POLICY,
    TTSConfig,
    TTSMessage,
    TTSRequestFactory,
    TTSRequestFactoryConfig,
)
from lalamo.safetensors import safe_read
from lalamo.sampling import SamplingPolicy

from .common import ParameterTree, unflatten_parameters

__all__ = ["TTSGenerationResult", "TTSGenerator", "TTSGeneratorConfig"]


@dataclass(frozen=True)
class TTSGeneratorConfig:
    tts_config: TTSConfig
    message_processor_config: TTSRequestFactoryConfig


@dataclass
class TTSGenerationResult:
    audio: np.ndarray
    audio_params: AudioRenderingSettings


class TTSGenerator(eqx.Module):
    config: TTSGeneratorConfig
    tts_model: TTSModel

    message_processor: TTSRequestFactory = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.tts_model.text_decoder.activation_precision

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "text_decoder": self.tts_model.text_decoder.export_weights(),
            "audio_decoder": self.tts_model.audio_decoder.export_weights(),
            "vocoder": self.tts_model.vocoder.export_weights(),
        }

    def tokenize_text(self, messages: Iterable[TTSMessage]) -> Int[Array, " batch tokens"]:
        text_tokens = self.message_processor.tokenize_request(messages)
        return jnp.asarray(text_tokens)[None, :]

    def decode_text(
        self,
        text_tokens: Array,
        sampling_policy: SamplingPolicy,
        repetition_penalty: float,  # noqa: ARG002, reserved for near future
        random_key: PRNGKeyArray | None = None,
    ) -> Array:
        random_key = jax.random.PRNGKey(123) if random_key is None else random_key
        return self.tts_model.text_decoder.decode_utterance(
            text_tokens,
            sampling_policy=sampling_policy,
            key=random_key,
        )

    def decode_audio(self, semantic_tokens: Array) -> Array:
        return self.tts_model.audio_decoder.audio_from_codes(semantic_tokens)

    def generate_waveform(self, audio_features: Array) -> Array:
        return self.tts_model.vocoder(audio_features)

    def get_generated_audio_params(self) -> AudioRenderingSettings:
        # NOTE: think if this could be moved to config level or made mandatory via abstract
        return AudioRenderingSettings(
            samplerate=DEFAULT_SAMPLERATE,
            output_channels=1,
            bitwidth=16,
            encoding=AudioEncoding.PCM,
        )

    def generate_speech(
        self,
        messages: Iterable[TTSMessage],
        sampling_policy: SamplingPolicy = DEFAULT_TTS_SAMPLING_POLICY,
        repetition_penalty: float = DEFAULT_TTS_REPETITION_PENALTY,
        random_key: PRNGKeyArray | None = None,
    ) -> TTSGenerationResult:
        text_tokens = self.tokenize_text(messages)

        semantic_tokens = self.decode_text(
            text_tokens,
            sampling_policy=sampling_policy,
            repetition_penalty=repetition_penalty,
            random_key=random_key,
        )

        audio_features = self.decode_audio(semantic_tokens)

        audio_waveform = self.tts_model.vocoder(audio_features)

        audio_settings = self.get_generated_audio_params()

        return TTSGenerationResult(audio=np.array(audio_waveform), audio_params=audio_settings)

    @classmethod
    def load_model(cls, path: Path | str) -> "TTSGenerator":
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)

        config = config_converter.structure(config_json["model_config"], TTSGeneratorConfig)
        assert isinstance(config, TTSGeneratorConfig)
        with Path(path / "model.safetensors").open("rb") as fd:
            _, weights_dict = safe_read(fd)
            weights = unflatten_parameters(weights_dict)
            model = config.tts_config.empty().import_weights(weights)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = TTSRequestFactory(config.message_processor_config, tokenizer)
        return TTSGenerator(
            config=config,
            tts_model=model,
            message_processor=message_processor,
        )


class FishAudioTTSGenerator(TTSGenerator):
    def decode_text(
        self,
        text_tokens: Int[Array, "batch sequence"],
        sampling_policy: SamplingPolicy | None = None,
        repetition_penalty: float = FishaudioConsts.DEFAULT_FISH_AUDIO_REPETITION_PENALTY,  # noqa: ARG002, reserved for near future
        random_key: PRNGKeyArray | None = None,
    ) -> Int[Array, "num_codebooks sequence"]:
        assert isinstance(self.tts_model.text_decoder, FishAudioTextDecoder)

        sampling_policy = sampling_policy if sampling_policy is not None else default_fishaudio_sampling_policy()

        random_key = jax.random.PRNGKey(DEFAULT_FISHAUDIO_RANDOM_SEED) if random_key is None else random_key
        return self.tts_model.text_decoder.decode_utterance(
            text_tokens,
            sampling_policy=sampling_policy,
            key=random_key,
        )

    def decode_audio(
        self,
        semantic_tokens: Int[Array, "batch n_codebooks sequence"],
    ) -> Float[Array, "batch audio_samples 1"]:
        return super().decode_audio(semantic_tokens)

    def generate_waveform(self, audio_features: Array) -> Array:
        return super().generate_waveform(audio_features)

    def get_generated_audio_params(self) -> AudioRenderingSettings:
        return AudioRenderingSettings(
            samplerate=self.tts_model.audio_decoder.samplerate,
            output_channels=1,
            bitwidth=16,
            encoding=AudioEncoding.PCM,
        )
