from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Array, DTypeLike, PRNGKeyArray

from lalamo.modules.common import LalamoModule
from lalamo.sampling import SamplingPolicy


@dataclass(frozen=True)
class TTSTextDecoderConfigBase(ABC):
    @abstractmethod
    def empty(self) -> "TTSTextDecoder": ...

    @abstractmethod
    def random_init(self, *, key: PRNGKeyArray) -> "TTSTextDecoder": ...


class TTSTextDecoder[ConfigT](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @abstractmethod
    def decode_utterance(
        self,
        text_tokens: Array,
        sampling_policy: SamplingPolicy | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Array: ...
