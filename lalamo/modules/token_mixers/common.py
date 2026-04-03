from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.modules.common import Initializer, LalamoModule, ParameterTree, PositionalEmbeddingSelector
from lalamo.modules.rope import PositionalEmbeddings

from .state.common import StateLayerBase

__all__ = [
    "MixerForwardPassConfig",
    "TokenMixerBase",
    "TokenMixerConfigBase",
    "TokenMixerResult",
]


@dataclass(frozen=True)
class MixerForwardPassConfig:
    pass


class TokenMixerResult[StateLayerT](NamedTuple):
    outputs: Float[Array, "*batch suffix_tokens channels"]
    state: StateLayerT | None = None


@dataclass(frozen=True)
class TokenMixerConfigBase(ABC):
    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "TokenMixerBase": ...


class TokenMixerBase[ConfigT, StateLayerT: StateLayerBase](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @abstractmethod
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: StateLayerT | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig | None = None,
    ) -> TokenMixerResult[StateLayerT]: ...

    @abstractmethod
    def init_static_state(self, capacity: int) -> StateLayerT: ...
