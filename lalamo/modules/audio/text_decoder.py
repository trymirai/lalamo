from abc import abstractmethod
from typing import Any, Self

from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterTree
from lalamo.modules.common import LalamoModule


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
        sampling_policy: Any | None = None,
        key: Any | None = None,
    ) -> Array: ...
