from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterTree
from lalamo.modules.common import Initializer, LalamoModule


@dataclass(frozen=True)
class TTSAudioDecoderConfigBase(ABC):
    @abstractmethod
    def init(self, initializer: Initializer) -> "TTSAudioDecoder": ...


class TTSAudioDecoder(LalamoModule):
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
