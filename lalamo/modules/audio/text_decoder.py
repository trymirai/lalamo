from abc import abstractmethod
from collections.abc import Callable
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

    @staticmethod
    def resolve(
        *,
        message_speaker: str | None,
        message_style: str | None,
        message_language: str | None,
        config: "TTSTextDecoderConfigBase",
        tokenize: Callable[[str], list[int]],
    ) -> "TTSDecodingContext":
        import jax.numpy as jnp

        speaker = message_speaker or config.default_speaker
        style = message_style or config.default_style
        language = message_language or config.default_language

        instruction_text = config.format_instruction(style) if style is not None else None
        instruction_tokens = jnp.asarray(tokenize(instruction_text))[None, :] if instruction_text is not None else None

        return TTSDecodingContext(
            speaker=speaker,
            language=language,
            instruction_tokens=instruction_tokens,
        )


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
    @abstractmethod
    def format_instruction(cls, style: str) -> str | None: ...


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
