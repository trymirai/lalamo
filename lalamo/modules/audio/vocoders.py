from abc import abstractmethod
from dataclasses import dataclass

from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.utils.registry_abc import RegistryABC


@dataclass(frozen=True)
class VocoderConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, initializer: Initializer) -> "Vocoder": ...


class Vocoder[ConfigT: VocoderConfig](LalamoModule[ConfigT]):
    @abstractmethod
    def __call__(
        self,
        audio_features: Float[Array, " audio_samples"],
    ) -> Float[Array, " audio_samples"]: ...


@dataclass(frozen=True)
class NoopVocoderConfig(VocoderConfig):
    def init(self, initializer: Initializer) -> "NoopVocoder":
        return NoopVocoder(
            config=self,
            sharding_config=initializer.sharding_config,
        )


class NoopVocoder(Vocoder[NoopVocoderConfig]):
    def __call__(
        self,
        audio_features: Float[Array, " audio_samples"],
    ) -> Float[Array, " audio_samples"]:
        return audio_features
