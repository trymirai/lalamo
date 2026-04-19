from abc import abstractmethod
from dataclasses import dataclass

from jaxtyping import Array, Key

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.sampling import SamplingPolicy
from lalamo.utils.registry_abc import RegistryABC


@dataclass(frozen=True)
class TTSTextDecoderConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, initializer: Initializer) -> "TTSTextDecoder": ...


class TTSTextDecoder[ConfigT: TTSTextDecoderConfig](LalamoModule[ConfigT]):
    @abstractmethod
    def decode_utterance(
        self,
        text_tokens: Array,
        sampling_policy: SamplingPolicy | None = None,
        *,
        key: Key[Array, ""],
        dequant_key: Key[Array, ""],
    ) -> Array: ...
