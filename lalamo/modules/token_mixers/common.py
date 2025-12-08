from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Self

from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.modules.common import LalamoModule, ParameterTree, PositionalEmbeddingSelector
from lalamo.modules.rope import PositionalEmbeddings

from .state.common import StateLayerBase

__all__ = [
    "TokenMixerBase",
    "TokenMixerConfigBase",
    "TokenMixerResult",
]


class TokenMixerResult[StateLayerT](NamedTuple):
    outputs: Float[Array, "*batch suffix_tokens channels"]
    state: StateLayerT | None = None


@dataclass(frozen=True)
class TokenMixerConfigBase(ABC):
    @property
    @abstractmethod
    def rope_dim(self) -> int | None: ...

    @abstractmethod
    def random_init(
        self,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "TokenMixerBase": ...

    @abstractmethod
    def empty(
        self,
        model_dim: int,
    ) -> "TokenMixerBase": ...


class TokenMixerBase[ConfigT, StateLayerT: StateLayerBase](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @property
    @abstractmethod
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector: ...

    @abstractmethod
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: StateLayerT | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> TokenMixerResult[StateLayerT]: ...

    @abstractmethod
    def init_static_state(self, capacity: int) -> StateLayerT: ...

    @abstractmethod
    def export_weights(self) -> ParameterTree: ...

    @abstractmethod
    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self: ...
