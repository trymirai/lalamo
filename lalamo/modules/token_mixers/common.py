from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import NamedTuple

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import MatmulConfig

from .state.common import StateLayerBase

__all__ = [
    "AttentionForwardPassConfig",
    "AttentionImplementation",
    "MixerForwardPassConfig",
    "PositionalEmbeddingSelector",
    "TokenMixerBase",
    "TokenMixerConfig",
    "TokenMixerResult",
]


class AttentionImplementation(Enum):
    STABLE_REDUCTION = "stable_reduction"
    STANDARD = "standard"
    CUDNN = "cudnn"
    TOKAMAX = "tokamax"


@dataclass(frozen=True)
class AttentionForwardPassConfig:
    implementation: AttentionImplementation = AttentionImplementation.STABLE_REDUCTION
    upcast_dtype: DTypeLike | None = jnp.float32
    arrays: MatmulConfig = field(default_factory=MatmulConfig)


MixerForwardPassConfig = AttentionForwardPassConfig


class TokenMixerResult[StateLayerT](NamedTuple):
    outputs: Float[Array, "*batch suffix_tokens channels"]
    state: StateLayerT | None = None


class PositionalEmbeddingSelector(StrEnum):
    GLOBAL = "global"
    LOCAL = "sliding_window"
    NONE = "none"


@dataclass(frozen=True)
class TokenMixerConfig(LalamoConfig, RegistryABC):
    @property
    @abstractmethod
    def rope_dim(self) -> int | None: ...

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "TokenMixerBase": ...


class TokenMixerBase[ConfigT: TokenMixerConfig, StateLayerT: StateLayerBase](LalamoModule[ConfigT]):
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
        dequant_key: Key[Array, ""],
    ) -> TokenMixerResult[StateLayerT]: ...

    @abstractmethod
    def init_static_state(self, capacity: int) -> StateLayerT: ...
