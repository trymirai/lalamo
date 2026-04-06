from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.modules.common import Initializer, LalamoModule, PositionalEmbeddingSelector
from lalamo.modules.forward_pass_config import ArrayForwardPassConfig, AttentionImplementation
from lalamo.modules.rope import PositionalEmbeddings

from .state.common import StateLayerBase

__all__ = [
    "AttentionForwardPassConfig",
    "MixerForwardPassConfig",
    "TokenMixerBase",
    "TokenMixerConfigBase",
    "TokenMixerResult",
]


@dataclass(frozen=True)
class AttentionForwardPassConfig:
    implementation: AttentionImplementation = AttentionImplementation.STABLE_REDUCTION
    upcast_dtype: DTypeLike | None = jnp.float32
    arrays: ArrayForwardPassConfig = field(default_factory=ArrayForwardPassConfig)


MixerForwardPassConfig = AttentionForwardPassConfig


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
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),  # noqa: B008
        *,
        key: PRNGKeyArray | None,
    ) -> TokenMixerResult[StateLayerT]: ...

    @abstractmethod
    def init_static_state(self, capacity: int) -> StateLayerT: ...
