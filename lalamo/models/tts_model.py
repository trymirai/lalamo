import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import DTypeLike, Int, PRNGKeyArray
from tokenizers import Tokenizer

from lalamo.audio.audio_rendering import AudioEncoding, AudioRenderingSettings
from lalamo.audio.tts_message_processor import (
    TTSMessage,
    TTSMessageProcessor,
    TTSMessageProcessorConfig,
)
from lalamo.modules import TTSModel, config_converter
from lalamo.modules.audio.text_decoder import TTSDecodingContext
from lalamo.modules.audio.text_to_speech import TTSConfig
from lalamo.safetensors import safe_read
from lalamo.sampling import SamplingPolicy

from .common import ParameterTree, unflatten_parameters

__all__ = ["TTSGenerationResult", "TTSGenerator", "TTSGeneratorConfig", "TTSMessage"]


@dataclass(frozen=True)
class TTSGeneratorConfig:
    tts_config: TTSConfig
    message_processor_config: TTSMessageProcessorConfig


@dataclass(frozen=True)
class TTSGenerationResult:
    audio: np.ndarray
    audio_params: AudioRenderingSettings


class TTSGenerator(eqx.Module):
    config: TTSGeneratorConfig
    tts_model: TTSModel

    message_processor: TTSMessageProcessor = eqx.field(static=True)

    @property
    def default_speaker(self) -> str | None:
        return self.tts_model.text_decoder.config.default_speaker

    @property
    def default_style(self) -> str | None:
        return self.tts_model.text_decoder.config.default_style

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
        random_key: PRNGKeyArray | None = None,
    ) -> TTSGenerationResult:
        messages_list = list(messages)
        if len(messages_list) != 1:
            raise ValueError(f"Expected exactly 1 message, got {len(messages_list)}")
        message = messages_list[0]

        config = self.tts_model.text_decoder.config
        speaker = message.speaker_id if message.speaker_id is not None else config.default_speaker
        style = message.style if message.style is not None else config.default_style
        language = message.language if message.language is not None else config.default_language

        instruction_text = config.format_instruction(style) if style is not None else None
        instruction_tokens = (
            jnp.asarray(self.message_processor.tokenize_text(instruction_text))[None, :]
            if instruction_text is not None
            else None
        )

        context = TTSDecodingContext(
            speaker=speaker,
            language=language,
            instruction_tokens=instruction_tokens,
        )

        text_tokens = self.tokenize_text(messages_list)
        random_key = random_key if random_key is not None else jax.random.key(123)

        codes = self.tts_model.text_decoder.decode_utterance(
            text_tokens,
            context=context,
            sampling_policy=sampling_policy,
            key=random_key,
        )

        audio_features = self.tts_model.audio_decoder.audio_from_codes(codes)
        audio_waveform = self.tts_model.vocoder(audio_features)

        return TTSGenerationResult(
            audio=np.array(audio_waveform),
            audio_params=self.get_generated_audio_params(),
        )

    @classmethod
    def load_model(cls, path: Path | str) -> "TTSGenerator":
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)

        config = config_converter.structure(config_json["model_config"], TTSGeneratorConfig)
        assert isinstance(config, TTSGeneratorConfig)
        with (path / "model.safetensors").open("rb") as fd:
            _, weights_dict = safe_read(fd)
            weights = unflatten_parameters(weights_dict)
            model = config.tts_config.empty().import_weights(weights)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = TTSMessageProcessor(config.message_processor_config, tokenizer)

        return TTSGenerator(config=config, tts_model=model, message_processor=message_processor)
