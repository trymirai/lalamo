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
        audio_features: Float[Array, "batch audio_channels samples"],
    ) -> Float[Array, "batch audio_channels samples"]: ...


@dataclass(frozen=True)
class NoopVocoderConfig(VocoderConfig):
    def init(self, initializer: Initializer) -> "NoopVocoder":  # noqa: ARG002
        return NoopVocoder(config=self)


class NoopVocoder(Vocoder[NoopVocoderConfig]):
    def __call__(
        self,
        audio_features: Float[Array, "batch audio_channels samples"],
    ) -> Float[Array, "batch audio_channels samples"]:
        return audio_features
