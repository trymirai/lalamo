import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jaxtyping import DTypeLike, PRNGKeyArray

from lalamo.audio.audio_rendering import AudioEncoding, AudioRenderingSettings
from lalamo.audio.tts_message_processor import (
    TTSMessage,
    TTSMessageProcessor,
    TTSMessageProcessorConfig,
)
from lalamo.modules import config_converter
from lalamo.modules.audio.latent_tts import (
    LatentTTSConfig,
    LatentTTSGenerationConfig,
    LatentTTSModel,
)
from lalamo.safetensors import safe_read

from .common import ParameterTree, unflatten_parameters
from .tts_model import TTSGenerationResult

__all__ = [
    "LatentTTSGenerator",
    "LatentTTSGeneratorConfig",
]


@dataclass(frozen=True)
class LatentTTSGeneratorConfig:
    latent_tts_config: LatentTTSConfig
    message_processor_config: TTSMessageProcessorConfig
    generation_config: LatentTTSGenerationConfig


class LatentTTSGenerator(eqx.Module):
    config: LatentTTSGeneratorConfig
    latent_tts_model: LatentTTSModel

    message_processor: TTSMessageProcessor = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.latent_tts_model.activation_precision

    def export_weights(self) -> ParameterTree[Array]:
        return self.latent_tts_model.export_weights()

    def get_generated_audio_params(self) -> AudioRenderingSettings:
        return AudioRenderingSettings(
            samplerate=self.latent_tts_model.samplerate,
            output_channels=1,
            bitwidth=16,
            encoding=AudioEncoding.PCM,
        )

    def generate_speech(
        self,
        messages: Iterable[TTSMessage],
        *,
        key: PRNGKeyArray | None = None,
        generation_config: LatentTTSGenerationConfig | None = None,
    ) -> TTSGenerationResult:
        if key is None:
            key = jax.random.PRNGKey(0)
        if generation_config is None:
            generation_config = self.config.generation_config

        output = self.latent_tts_model.generate_speech_from_messages(
            messages,
            self.message_processor,
            self.config.latent_tts_config,
            key=key,
            generation_config=generation_config,
        )

        waveform_length = int(output.waveform_lengths[0])
        waveform = np.array(output.waveform[0, :waveform_length], dtype=np.float32)

        return TTSGenerationResult(
            audio=waveform,
            audio_params=self.get_generated_audio_params(),
        )

    @classmethod
    def load_model(cls, path: Path | str) -> "LatentTTSGenerator":
        if isinstance(path, str):
            path = Path(path)

        with (path / "config.json").open() as config_file:
            config_json = json.load(config_file)

        config = config_converter.structure(config_json["model_config"], LatentTTSGeneratorConfig)
        assert isinstance(config, LatentTTSGeneratorConfig)

        weights_dict, _ = safe_read(path / "model.safetensors")
        weights = unflatten_parameters(weights_dict)
        model = config.latent_tts_config.empty().import_weights(weights)

        tokenizer = config.latent_tts_config.create_tokenizer(path)
        message_processor = config.latent_tts_config.create_message_processor(
            config.message_processor_config, tokenizer
        )

        return cls(config=config, latent_tts_model=model, message_processor=message_processor)
