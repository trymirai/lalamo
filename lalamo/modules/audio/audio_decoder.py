from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from jaxtyping import Array, DTypeLike, PRNGKeyArray

from lalamo.common import ParameterTree
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

    @abstractmethod
    def export_weights(self) -> ParameterTree[Array]: ...

    @abstractmethod
    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self: ...

    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @abstractmethod
    def audio_from_codes(self, indices: Array) -> Array: ...
