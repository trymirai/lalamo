from abc import abstractmethod
from dataclasses import dataclass

from jaxtyping import Array, DTypeLike, Key

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.utils.registry_abc import RegistryABC


@dataclass(frozen=True)
class TTSAudioDecoderConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, initializer: Initializer) -> "TTSAudioDecoder": ...


class TTSAudioDecoder[ConfigT: TTSAudioDecoderConfig](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @abstractmethod
    def audio_from_codes(self, indices: Array, *, dequant_key: Key[Array, ""]) -> Array: ...
