import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
from jax import Array
from jaxtyping import DTypeLike, PRNGKeyArray
from tokenizers import Tokenizer

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
        raise NotImplementedError(
            "LatentTTSGenerator.generate_speech() must be implemented by a subclass "
            "that knows how to prepare model-specific inputs (e.g. reference audio)."
        )

    @classmethod
    def load_model(cls, path: Path | str) -> "LatentTTSGenerator":
        if isinstance(path, str):
            path = Path(path)

        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)

        config = config_converter.structure(config_json["model_config"], LatentTTSGeneratorConfig)
        assert isinstance(config, LatentTTSGeneratorConfig)

        with (path / "model.safetensors").open("rb") as fd:
            _, weights_dict = safe_read(fd)
            weights = unflatten_parameters(weights_dict)
            model = config.latent_tts_config.empty().import_weights(weights)

        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = TTSMessageProcessor(config.message_processor_config, tokenizer)

        generator_cls = config.latent_tts_config.generator_class()
        return generator_cls(
            config=config,
            latent_tts_model=model,
            message_processor=message_processor,
        )
