from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Self

from jaxtyping import Array, DTypeLike, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.common import LalamoModule
from lalamo.registry_abc import RegistryABC
from lalamo.sampling import SamplingPolicy


class CodebookCodes(NamedTuple):
    semantic: Int[Array, "..."]
    acoustic: Int[Array, "..."]


@dataclass(frozen=True)
class TTSDecodingContext:
    speaker: str | None = None
    language: str | None = None
    instruction_tokens: Int[Array, "batch tokens"] | None = None


@dataclass(frozen=True, kw_only=True)
class TTSTextDecoderConfigBase(RegistryABC):
    default_speaker: str | None = None
    default_style: str | None = None
    default_language: str | None = None

    @abstractmethod
    def empty(self) -> "TTSTextDecoder": ...

    @abstractmethod
    def random_init(self, *, key: PRNGKeyArray) -> "TTSTextDecoder": ...

    @classmethod
    def format_instruction(cls, style: str) -> str | None:  # noqa: ARG003
        return None


class TTSTextDecoder[ConfigT](LalamoModule[ConfigT]):
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

    @abstractmethod
    def decode_utterance(
        self,
        text_tokens: Array,
        *,
        context: TTSDecodingContext,
        sampling_policy: SamplingPolicy | None = None,
        key: PRNGKeyArray,
    ) -> CodebookCodes: ...
