import json
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
from tokenizers import Tokenizer

from lalamo.audio import AudioEncoding, AudioRenderer, AudioRenderingConfig
from lalamo.model_import.loaders.fishaudio_loaders import load_descript_audio_codec
from lalamo.model_import.model_specs.common import cast_if_float
from lalamo.modules import NoopVocoder, TTSModel, VocoderConfig, config_converter
from lalamo.modules.audio.fishaudio import DescriptAudioCodec, DescriptAudioCodecConfig, FishAudioTextDecoderConfig
from lalamo.modules.audio.fishaudio.fishaudio_sampling import (
    DEFAULT_FISH_AUDIO_REPETITION_PENALTY,
    DEFAULT_FISH_AUDIO_SAMPLING_POLICY,
    sampling_params_from_policy,
)
from lalamo.modules.audio.fishaudio.fishaudio_text_decoding import (
    FishAudioTextDecoder,
    load_tokenizer_from_fish_audio,
)
from lalamo.modules.audio.fishaudio.fishaudio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
    FishAudioTextDecoderConfig_Foreign,
    load_fish_audio_audio_decoder,
    try_locate_fish_audio_model_path,
)
from lalamo.modules.audio.text_to_speech import (
    DEFAULT_TTS_REPETITION_PENALTY,
    DEFAULT_TTS_SAMPLING_POLICY,
    TTSConfig,
    TTSMessage,
    TTSRequestFactory,
    TTSRequestFactoryConfig,
)
from lalamo.modules.audio.utils import DTypeConvert
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.safetensors import safe_read
from lalamo.sampling import SamplingPolicy
from lalamo.utils import MapDictValues

from .common import ParameterTree, unflatten_parameters

__all__ = ["ForeignTTSModelType", "TTSGenerationResult", "TTSGenerator", "TTSGeneratorConfig"]


class ForeignTTSModelType(Enum):
    FISH_AUDIO = "fishaudio"
    FISH_AUDIO_LALAMO = "fishaudio_lalamo"


@dataclass(frozen=True)
class TTSGeneratorConfig:
    tts_config: TTSConfig
    message_processor_config: TTSRequestFactoryConfig


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
            "text_decoder": self.tts_model.text_decoder.export_weights(),
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
    def decode_text(
        self,
        text_tokens: Array,
        sampling_policy: SamplingPolicy,
        repetition_penalty: float,
        random_key: PRNGKeyArray | None = None,
    ) -> Array: ...

    @abstractmethod
    def decode_audio(self, semantic_tokens: Array) -> Array:
        return self.tts_model.audio_decoder.audio_from_codes(semantic_tokens)

    @abstractmethod
    def generate_waveform(self, audio_features: Array) -> Array:
        return self.tts_model.vocoder(audio_features)

    @abstractmethod
    def get_generated_audio_params(self) -> AudioRenderingConfig:
        # NOTE: mb this could be moved to config level
        return AudioRenderingConfig(
            samplerate=self.tts_model.audio_decoder.samplerate,
            output_channels=1,
            bitwidth=16,
            encoding=AudioEncoding.pcm,
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
            text_tokens, sampling_policy=sampling_policy, repetition_penalty=repetition_penalty, random_key=random_key
        )

        audio_features = self.decode_audio(semantic_tokens)

        audio_waveform = self.tts_model.vocoder(audio_features)

        audio_waveform = self.audio_renderer.condition_signal(
            generated_audio=np.array(audio_waveform),
            generated_audio_properties=self.get_generated_audio_params(),
        )

        return TTSGenerationResult(audio=audio_waveform, audio_params=self.audio_renderer.config)


@dataclass(frozen=True)
class FishAudioGeneratorConfig(TTSGeneratorConfig):
    tts_config: TTSConfig[FishAudioTextDecoderConfig, DescriptAudioCodecConfig]
    message_processor_config: TTSRequestFactoryConfig


class FishAudioTTSGenerator(TTSGenerator):
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
        return super().decode_audio(semantic_tokens)

    def generate_waveform(self, audio_features: Array) -> Array:
        return super().generate_waveform(audio_features)

    def get_generated_audio_params(self) -> AudioRenderingConfig:
        return super().get_generated_audio_params()


def _build_foreign_fish_audio_tts_generator(
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
    return FishAudioTTSGenerator(
        config=FishAudioGeneratorConfig(tts_config=tts_config, message_processor_config=message_processor.config),
        tts_model=tts_model,
        message_processor=message_processor,
        audio_renderer=AudioRenderer(audio_renderer_config),
    )


def _build_lalamo_fish_audio_tts_generator_from_checkpoint(
    path_to_checkpoints: Path, device="cpu", precision: torch.dtype = torch.bfloat16
) -> "TTSGenerator":
    path_to_audio_model = path_to_checkpoints / "codec.pth"
    text_decoder = FishAudioModeling.text_decoder_from_foreign_model(path_to_checkpoints, jnp.bfloat16)
    audio_decoder = FishAudioModeling.dac_from_foreign_model(path_to_audio_model, jnp.float32)

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
    return FishAudioTTSGenerator(
        config=FishAudioGeneratorConfig(tts_config=tts_config, message_processor_config=message_processor.config),
        message_processor=message_processor,
        tts_model=tts_model,
        audio_renderer=AudioRenderer(audio_renderer_config),
    )


class FishAudioModeling:
    from fish_speech.models.text2semantic.llama import DualARTransformer

    @staticmethod
    def dac_from_foreign_model(audio_chkpt_path: Path, precision: DTypeLike) -> DescriptAudioCodec:
        from fish_speech.models.dac.inference import load_model
        from fish_speech.models.dac.modded_dac import DAC

        fish_dac = load_model("modded_dac_vq", audio_chkpt_path, device="cpu")
        assert isinstance(fish_dac, DAC)

        dac_weights = dict(
            MapDictValues(
                lambda v: cast_if_float(torch_to_jax(v), precision),
                fish_dac.state_dict(),
            )
        )

        return load_descript_audio_codec(dac_weights)

    @staticmethod
    def text_decoder_from_foreign_model(
        fish_model_or_path: Path | DualARTransformer, precision: DTypeLike
    ) -> FishAudioTextDecoder:
        from lalamo.model_import.loaders.fishaudio_loaders import load_fishaudio_text_decoder

        return load_fishaudio_text_decoder(fish_model_or_path, precision)


@dataclass(frozen=True)
class TTSLoader:
    @staticmethod
    def load_model_from_foreign_model_preset(preset: ForeignTTSModelType, path_to_checkpoints: Path) -> "TTSGenerator":
        match preset:
            case ForeignTTSModelType.FISH_AUDIO:
                return _build_foreign_fish_audio_tts_generator(path_to_checkpoints)
            case ForeignTTSModelType.FISH_AUDIO_LALAMO:
                return _build_lalamo_fish_audio_tts_generator_from_checkpoint(path_to_checkpoints)

    @staticmethod
    def try_locate_audio_model_path(preset: ForeignTTSModelType) -> Path | None:
        match preset:
            case ForeignTTSModelType.FISH_AUDIO | ForeignTTSModelType.FISH_AUDIO_LALAMO:
                return try_locate_fish_audio_model_path()

    @staticmethod
    def tts_generator_type_from_model_name(tts_model_type: str):
        if tts_model_type == "FishAudio/openaudio/openaudio-s1-mini":
            return FishAudioTTSGenerator, FishAudioGeneratorConfig
        else:
            raise ValueError(f"Unsupported TTS model: {tts_model_type}")

    @staticmethod
    def load_model(path: Path | str) -> TTSGenerator:
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)
        full_tts_model_name: str = f"{config_json['vendor']}/{config_json['family']}/{config_json['name']}"
        generator_type, generator_config_type = TTSLoader.tts_generator_type_from_model_name(full_tts_model_name)

        config = config_converter.structure(config_json["model_config"], generator_config_type)
        with Path(path / "model.safetensors").open("rb") as fd:
            _, weights_dict = safe_read(fd)
            weights = unflatten_parameters(weights_dict)
            model = config.tts_config.empty().import_weights(weights)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = TTSRequestFactory(config.message_processor_config, tokenizer)
        audio_renderer_config = AudioRenderingConfig(
            config.tts_config.audio_decoder_config.samplerate, 1, 16, AudioEncoding.pcm
        )
        return generator_type(
            config=config,
            tts_model=model,
            audio_renderer=AudioRenderer(config=audio_renderer_config),
            message_processor=message_processor,
        )
