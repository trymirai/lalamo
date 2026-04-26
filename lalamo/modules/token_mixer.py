from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import NamedTuple, Self

import equinox as eqx
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import MatmulConfig

from .rope import PositionalEmbeddings

__all__ = [
    "AttentionImplementation",
    "MixerForwardPassConfig",
    "PositionalEmbeddingSelector",
    "State",
    "StateLayerBase",
    "TokenMixerBase",
    "TokenMixerConfig",
    "TokenMixerResult",
]


class StateLayerBase(Exportable, eqx.Module):
    pass


@register_pytree_node_class
class State(tuple[StateLayerBase, ...]):
    __slots__ = ()

    def tree_flatten(self) -> tuple[tuple[StateLayerBase, ...], None]:
        return (tuple(self), None)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[StateLayerBase, ...]) -> Self:  # noqa: ARG003
        return cls(children)


class AttentionImplementation(Enum):
    STABLE_REDUCTION = "stable_reduction"
    STANDARD = "standard"
    CUDNN = "cudnn"
    TOKAMAX = "tokamax"


@dataclass(frozen=True)
class MixerForwardPassConfig:
    implementation: AttentionImplementation = AttentionImplementation.STABLE_REDUCTION
    upcast_dtype: DTypeLike | None = jnp.float32
    arrays: MatmulConfig = field(default_factory=MatmulConfig)


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
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[StateLayerT]: ...

    @abstractmethod
    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StateLayerT: ...
