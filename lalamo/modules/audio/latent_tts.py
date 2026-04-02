from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Array, Float, Int, PRNGKeyArray
from tokenizers import Tokenizer

from lalamo.audio.tts_message_processor import TTSMessage, TTSMessageProcessor, TTSMessageProcessorConfig
from lalamo.modules.common import LalamoModule
from lalamo.registry_abc import RegistryABC


@dataclass(frozen=True)
class LatentTTSOutputs:
    waveform: Float[Array, "batch audio_samples"]
    waveform_lengths: Int[Array, " batch"]


@dataclass(frozen=True)
class LatentTTSGenerationConfig:
    n_timesteps: int
    temperature: float
    length_scale: float
    guidance_scale: float


class LatentTTSModel[ConfigT](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401

    @abstractmethod
    def predict(
        self,
        encoded: Any,  # noqa: ANN401
        *,
        key: PRNGKeyArray,
        generation_config: LatentTTSGenerationConfig,
    ) -> Any: ...  # noqa: ANN401

    @abstractmethod
    def decode(self, predicted: Any) -> LatentTTSOutputs: ...  # noqa: ANN401

    @abstractmethod
    def generate_speech_from_messages(
        self,
        messages: Iterable[TTSMessage],
        message_processor: TTSMessageProcessor,
        latent_tts_config: "LatentTTSConfig",
        *,
        key: PRNGKeyArray,
        generation_config: LatentTTSGenerationConfig,
    ) -> LatentTTSOutputs: ...


class LatentTTSConfig(RegistryABC):
    @abstractmethod
    def empty(self) -> LatentTTSModel: ...

    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @abstractmethod
    def default_generation_config(self) -> LatentTTSGenerationConfig: ...

    def create_tokenizer(self, model_path: Path | str) -> Tokenizer:
        return Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))

    def create_message_processor(self, config: TTSMessageProcessorConfig, tokenizer: Tokenizer) -> TTSMessageProcessor:
        return TTSMessageProcessor(config, tokenizer)
