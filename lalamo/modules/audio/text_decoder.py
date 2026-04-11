from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Array, Key

from lalamo.module import Initializer, LalamoModule
from lalamo.sampling import SamplingPolicy


@dataclass(frozen=True)
class TTSTextDecoderConfigBase(ABC):
    @abstractmethod
    def init(self, initializer: Initializer) -> "TTSTextDecoder": ...


class TTSTextDecoder[ConfigT: TTSTextDecoderConfigBase](LalamoModule[ConfigT]):
    @abstractmethod
    def decode_utterance(
        self,
        text_tokens: Array,
        sampling_policy: SamplingPolicy | None = None,
        key: Key[Array, ""] | None = None,
    ) -> Array: ...
