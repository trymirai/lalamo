from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from jaxtyping import Array, Float, Int, PRNGKeyArray

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


class LatentTTSConfig(RegistryABC):
    @abstractmethod
    def empty(self) -> LatentTTSModel: ...

    @property
    @abstractmethod
    def samplerate(self) -> int: ...
