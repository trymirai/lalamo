from abc import abstractmethod
from dataclasses import dataclass

from jaxtyping import Array

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.utils.registry_abc import RegistryABC


@dataclass(frozen=True)
class TTSAudioDecoderConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, initializer: Initializer) -> "TTSAudioDecoder": ...


class TTSAudioDecoder[ConfigT: TTSAudioDecoderConfig](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @abstractmethod
    def audio_from_codes(self, indices: Array, *, keychain: Keychain) -> Array: ...
