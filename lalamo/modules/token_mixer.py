from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Self

import equinox as eqx
import jax
from jax import numpy as jnp
from jax.lax import DotAlgorithmPreset
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.normalization import NormalizationForwardPassConfig
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import GradientEstimator, MatmulConfig

from .rope import PositionalEmbeddings

__all__ = [
    "AttentionImplementation",
    "MixerForwardPassConfig",
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
    attention_implementation: AttentionImplementation = AttentionImplementation.STANDARD
    attention_accumulation_dtype: DTypeLike | None = jnp.float32
    rope_dtype: DTypeLike | None = jnp.float32
    attention_tile_size: int = 128
    ssm_chunk_size: int = 32
    ssm_min_tail_size_to_chunk: int = 16
    matmul_config: MatmulConfig = field(default_factory=MatmulConfig)
    normalization_forward_pass_config: NormalizationForwardPassConfig = field(
        default_factory=NormalizationForwardPassConfig,
    )

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            attention_implementation=AttentionImplementation.STABLE_REDUCTION,
            matmul_config=MatmulConfig.for_tracer_tests(),
            normalization_forward_pass_config=NormalizationForwardPassConfig.for_tracer_tests(),
        )

    @classmethod
    def for_inference(cls, precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT) -> Self:
        if jax.default_backend() == "cpu":
            attention_implementation = AttentionImplementation.STANDARD
        else:
            attention_implementation = AttentionImplementation.CUDNN
        return cls(
            attention_implementation=attention_implementation,
            matmul_config=MatmulConfig.for_inference(precision),
            normalization_forward_pass_config=NormalizationForwardPassConfig.for_inference(),
        )

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            attention_implementation=AttentionImplementation.STANDARD,
            ssm_min_tail_size_to_chunk=0,
            matmul_config=MatmulConfig.for_training(gradient_estimator, precision),
            normalization_forward_pass_config=NormalizationForwardPassConfig.for_training(),
        )


class TokenMixerResult[StateLayerT](NamedTuple):
    outputs: Float[Array, "*batch suffix_tokens channels"]
    state: StateLayerT | None = None
    positional_embeddings: PositionalEmbeddings | None = None


@dataclass(frozen=True)
class TokenMixerConfig(LalamoConfig, RegistryABC):
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
        token_positions: Int[Array, " suffix_tokens"],
        state: StateLayerT | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[StateLayerT]: ...

    @abstractmethod
    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StateLayerT: ...
