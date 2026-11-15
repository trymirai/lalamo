from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.utils.registry_abc import RegistryABC

from .activations import Activation
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .residual import AdditiveResidualConfig, ResidualConfig
from .token_mixer import StateLayerBase
from .utils import call_vmapped_twice

__all__ = [
    "PerLayerEmbeddingInjection",
    "PerLayerEmbeddingInjectionConfig",
    "ResidualScale",
    "ResidualScaleConfig",
    "StreamOpBase",
    "StreamOpConfig",
    "StreamOpContext",
]


class StreamOpContext(eqx.Module):
    token_ids: Int[Array, "batch suffix_tokens"] | None
    per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None
    lengths_without_padding: Int[Array, " batch"] | None


@dataclass(frozen=True)
class StreamOpConfig(LalamoConfig, RegistryABC):
    @property
    @abstractmethod
    def is_token_local(self) -> bool: ...

    @property
    @abstractmethod
    def is_stateful(self) -> bool: ...

    @abstractmethod
    def init(self, initializer: Initializer, residual_config: ResidualConfig, model_dim: int) -> "StreamOpBase": ...


class StreamOpBase[ConfigT: StreamOpConfig, StreamT, StateT: StateLayerBase](LalamoModule[ConfigT]):
    @abstractmethod
    def __call__(
        self,
        stream: StreamT,
        context: StreamOpContext,
        state: StateT | None,
        *,
        keychain: Keychain,
    ) -> tuple[StreamT, StateT | None]: ...

    @abstractmethod
    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StateT | None: ...


@dataclass(frozen=True)
class PerLayerEmbeddingInjectionConfig(StreamOpConfig):
    linear_config: LinearConfig
    norm_config: NormalizationConfig
    ple_dim: int
    activation: Activation

    @property
    def is_token_local(self) -> bool:
        return True

    @property
    def is_stateful(self) -> bool:
        return False

    def init(
        self,
        initializer: Initializer,
        residual_config: ResidualConfig,
        model_dim: int,
    ) -> "PerLayerEmbeddingInjection":
        if not isinstance(residual_config, AdditiveResidualConfig):
            raise TypeError(
                f"Per-layer embedding injection requires an additive residual, got {type(residual_config).__name__}.",
            )
        gate = self.linear_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(self.ple_dim,),
            has_biases=False,
        )
        projection = self.linear_config.init(
            initializer,
            input_dim=self.ple_dim,
            output_dims=(model_dim,),
            has_biases=False,
        )
        norm = self.norm_config.init(initializer, model_dim)
        return PerLayerEmbeddingInjection(
            config=self,
            sharding_config=initializer.sharding_config,
            gate=gate,
            projection=projection,
            norm=norm,
        )


class PerLayerEmbeddingInjection(
    StreamOpBase[PerLayerEmbeddingInjectionConfig, Float[Array, "batch suffix_tokens channels"], StateLayerBase],
):
    gate: Linear
    projection: Linear
    norm: Normalization

    def __call__(
        self,
        stream: Float[Array, "batch suffix_tokens channels"],
        context: StreamOpContext,
        state: StateLayerBase | None,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens channels"], StateLayerBase | None]:
        if context.per_layer_input is None:
            raise ValueError("Per-layer embedding injection requires per_layer_input.")
        gate_keychain, projection_keychain = keychain.split()
        (ple_gated,) = call_vmapped_twice(
            self.gate,
            stream,
            keychain=gate_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        ple_gated = self.config.activation(ple_gated) * context.per_layer_input
        (ple_projected,) = call_vmapped_twice(
            self.projection,
            ple_gated,
            keychain=projection_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        ple_normed = call_vmapped_twice(self.norm, ple_projected)
        return stream + ple_normed, state

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StateLayerBase | None:  # noqa: ARG002
        return None


@dataclass(frozen=True)
class ResidualScaleConfig(StreamOpConfig):
    @property
    def is_token_local(self) -> bool:
        return True

    @property
    def is_stateful(self) -> bool:
        return False

    def init(
        self,
        initializer: Initializer,
        residual_config: ResidualConfig,
        model_dim: int,  # noqa: ARG002
    ) -> "ResidualScale":
        if not isinstance(residual_config, AdditiveResidualConfig):
            raise TypeError(f"Residual scaling requires an additive residual, got {type(residual_config).__name__}.")
        return ResidualScale(
            config=self,
            sharding_config=initializer.sharding_config,
            scale=initializer.ones((1,)),
        )


class ResidualScale(
    StreamOpBase[ResidualScaleConfig, Float[Array, "batch suffix_tokens channels"], StateLayerBase],
):
    scale: Float[Array, "1"]

    def __call__(
        self,
        stream: Float[Array, "batch suffix_tokens channels"],
        context: StreamOpContext,  # noqa: ARG002
        state: StateLayerBase | None,
        *,
        keychain: Keychain,  # noqa: ARG002
    ) -> tuple[Float[Array, "batch suffix_tokens channels"], StateLayerBase | None]:
        return stream * self.scale, state

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StateLayerBase | None:  # noqa: ARG002
        return None
