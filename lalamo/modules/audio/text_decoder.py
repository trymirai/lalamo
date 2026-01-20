from abc import abstractmethod
from typing import Self

from jaxtyping import Array, DTypeLike, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.common import LalamoModule
from lalamo.sampling import SamplingPolicy


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
        sampling_policy: SamplingPolicy | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Array: ...
