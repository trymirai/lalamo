from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Array, DTypeLike, PRNGKeyArray

from lalamo.modules.common import LalamoModule


@dataclass(frozen=True)
class TTSAudioDecoderConfigBase(ABC):
    @abstractmethod
    def empty(self) -> "TTSAudioDecoder": ...

    @abstractmethod
    def random_init(self, *, key: PRNGKeyArray) -> "TTSAudioDecoder": ...


class TTSAudioDecoder[ConfigT](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @abstractmethod
    def audio_from_codes(self, indices: Array) -> Array: ...
