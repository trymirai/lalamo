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
    # DeltaNet chunked attention parameters.
    chunk_size: int = 32
    # Minimum tokens in the last chunk before it falls back to sequential scan.
    # If the remainder after chunking is shorter than this, the tail is processed
    # sequentially to avoid wasting compute on mostly-padded chunks.
    min_chunk_len: int = 16


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


class TokenMixerBase[StateLayerT: StateLayerBase](LalamoModule):
    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

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
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),  # noqa: B008
    ) -> TokenMixerResult[StateLayerT]: ...

    @abstractmethod
    def init_static_state(self, capacity: int) -> StateLayerT: ...
