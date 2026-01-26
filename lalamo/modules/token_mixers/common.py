from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from jaxtyping import Array, Float, Int

from lalamo.common import RegistryABC
from lalamo.modules.common import Initializer, LalamoConfig, LalamoModule, PositionalEmbeddingSelector
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
class TokenMixerConfigBase(LalamoConfig, RegistryABC):
    @property
    @abstractmethod
    def rope_dim(self) -> int | None: ...

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "TokenMixerBase": ...


class TokenMixerBase[StateLayerT: StateLayerBase](LalamoModule):
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
