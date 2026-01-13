from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

import equinox as eqx
import jax
import numpy as np
import torch
from jax import Array
from jax import numpy as jnp
from jaxtyping import DTypeLike, Int, PRNGKeyArray

from lalamo.audio import AudioEncoding, AudioRenderer, AudioRenderingConfig
from lalamo.modules import NoopVocoder, TTSModel, VocoderConfig
from lalamo.modules.audio.foreign.fishaudio_audio_decoding import DescriptAudioCodecConfig
from lalamo.modules.audio.foreign.fishaudio_sampling import (
    DEFAULT_FISH_AUDIO_REPETITION_PENALTY,
    DEFAULT_FISH_AUDIO_SAMPLING_POLICY,
    sampling_params_from_policy,
)
from lalamo.modules.audio.foreign.fishaudio_text_decoding import (
    FishAudioTextDecoder,
    FishAudioTextDecoderConfig,
    load_tokenizer_from_fish_audio,
)
from lalamo.modules.audio.foreign.fishaudio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
    FishAudioTextDecoderConfig_Foreign,
    load_fish_audio_audio_decoder,
    try_locate_fish_audio_model_path,
)
from lalamo.modules.audio.text_to_speech import TTSConfig, TTSMessage, TTSRequestFactory, TTSRequestFactoryConfig
from lalamo.modules.audio.utils import DTypeConvert
from lalamo.sampling import SamplingPolicy

from .common import ParameterTree

__all__ = ["ForeignTTSModelType", "TTSGenerationResult", "TTSGenerator", "TTSGeneratorConfig"]


class ForeignTTSModelType(Enum):
    FISH_AUDIO = "fishaudio"
    FISH_AUDIO_LALAMO = "fishaudio_lalamo"


@dataclass(frozen=True)
class TTSGeneratorConfig:
    tts_config: TTSConfig

    @classmethod
    def load_model_from_foreign_model_preset(
        cls, preset: ForeignTTSModelType, path_to_checkpoints: Path
    ) -> "TTSGenerator":
        match preset:
            case ForeignTTSModelType.FISH_AUDIO:
                return _build_foreign_fish_audio_model(path_to_checkpoints)
            case ForeignTTSModelType.FISH_AUDIO_LALAMO:
                return _build_lalamo_model_from_fish_checkpoint(path_to_checkpoints)

    @classmethod
    def try_locate_audio_model_path(cls, preset: ForeignTTSModelType) -> Path | None:
        match preset:
            case ForeignTTSModelType.FISH_AUDIO | ForeignTTSModelType.FISH_AUDIO_LALAMO:
                return try_locate_fish_audio_model_path()


@dataclass
class TTSGenerationResult:
    audio: np.ndarray
    audio_params: AudioRenderingConfig


@dataclass
class TTSGenerator:
    config: TTSGeneratorConfig

    tts_model: TTSModel

    audio_renderer: AudioRenderer = eqx.field(static=True)
    message_processor: TTSRequestFactory = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.tts_model.text_decoder.activation_precision

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "text_encoder": self.tts_model.text_decoder.export_weights(),
            "audio_decoder": self.tts_model.audio_decoder.export_weights(),
            "vocoder": self.tts_model.vocoder.export_weights(),
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

        audio_features = self.decode_audio(semantic_tokens)

        audio_waveform = self.tts_model.vocoder(audio_features)

        audio_waveform = self.audio_renderer.condition_signal(
            generated_audio=np.array(audio_waveform),
            generated_audio_properties=self.get_generated_audio_params(),
        )

        return TTSGenerationResult(audio=audio_waveform, audio_params=self.audio_renderer.config)


class FishAudioTTSGenerator_Foreign(TTSGenerator):
    def decode_text(
        self,
        text_tokens: Array,
        sampling_policy: SamplingPolicy = DEFAULT_FISH_AUDIO_SAMPLING_POLICY,
        repetition_penalty: float = DEFAULT_FISH_AUDIO_REPETITION_PENALTY,
        random_key: PRNGKeyArray | None = None,
    ) -> Array:
        assert isinstance(self.tts_model.text_decoder, (FishAudioTextDecoder_Foreign, FishAudioTextDecoder))

        sampling_params = sampling_params_from_policy(sampling_policy, repetition_penalty)
        random_key = jax.random.PRNGKey(123) if random_key is None else random_key
        return self.tts_model.text_decoder.decode_utterance(
            text_tokens, sampling_params=sampling_params, key=random_key
        )

    def decode_audio(self, semantic_tokens: Array) -> Array:
        return self.tts_model.audio_decoder.audio_from_codes(semantic_tokens)

    def generate_waveform(self, audio_features: Array) -> Array:
        return self.tts_model.vocoder(audio_features)

    def get_generated_audio_params(self) -> AudioRenderingConfig:
        # NOTE: mb this could be moved to config level
        return AudioRenderingConfig(
            samplerate=self.tts_model.audio_decoder.samplerate,
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
        activation_precision=DTypeConvert.to_jax(precision),
    )

    audio_renderer_config = AudioRenderingConfig(44100, 1, 16, AudioEncoding.pcm)
    tts_model = TTSModel(
        config=tts_config,
        text_decoder=text_decoder,
        audio_decoder=audio_decoder,
        vocoder=NoopVocoder(tts_config.vocoder_config),
    )
    return FishAudioTTSGenerator_Foreign(
        config=TTSGeneratorConfig(tts_config=tts_config),
        tts_model=tts_model,
        message_processor=message_processor,
        audio_renderer=AudioRenderer(audio_renderer_config),
    )


def _build_lalamo_model_from_fish_checkpoint(
    path_to_checkpoints: Path, device="cpu", precision: torch.dtype = torch.bfloat16
) -> "TTSGenerator":
    path_to_audio_model = path_to_checkpoints / "codec.pth"
    text_decoder = FishAudioTextDecoderConfig.from_foreign_model(path_to_checkpoints, jnp.bfloat16)
    audio_decoder = DescriptAudioCodecConfig.from_foreign_model(path_to_audio_model, jnp.float32)

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
        activation_precision=DTypeConvert.to_jax(precision),
    )
    audio_renderer_config = AudioRenderingConfig(44100, 1, 16, AudioEncoding.pcm)
    tts_model = TTSModel(
        config=tts_config,
        text_decoder=text_decoder,
        audio_decoder=audio_decoder,
        vocoder=NoopVocoder(tts_config.vocoder_config),
    )
    return FishAudioTTSGenerator_Foreign(
        config=TTSGeneratorConfig(tts_config=tts_config),
        message_processor=message_processor,
        tts_model=tts_model,
        audio_renderer=AudioRenderer(audio_renderer_config),
    )
