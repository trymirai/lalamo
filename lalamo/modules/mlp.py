from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterTree

from .activations import Activation
from .common import DummyUnionMember, LalamoModule, WeightLayout, register_config_union
from .linear import LinearBase, LinearConfig

__all__ = [
    "DenseMLP",
    "DenseMLPConfig",
    "MLPBase",
    "MLPConfig",
    "MixtureOfExperts",
    "MixtureOfExpertsConfig",
    "RoutingFunction",
    "SoftmaxRouting",
]


@dataclass(frozen=True)
class DenseMLPConfig:
    linear_config: LinearConfig
    activation: Activation
    has_up_biases: bool
    has_down_biases: bool
    gate_clipping: tuple[float | None, float | None] | None
    up_clipping: tuple[float | None, float | None] | None

    def random_init(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> "DenseMLP":
        up_key, down_key = jax.random.split(key)
        return DenseMLP(
            self,
            up_projection=self.linear_config.random_init(
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
                key=up_key,
            ),
            down_projection=self.linear_config.random_init(
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
                key=down_key,
            ),
        )

    def empty(self, model_dim: int, hidden_dim: int) -> "DenseMLP":
        return DenseMLP(
            self,
            up_projection=self.linear_config.empty(
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
            ),
            down_projection=self.linear_config.empty(
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
            ),
        )

    def random_init_mixture(
        self,
        mixture_size: int,
        model_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "DenseMLP":
        up_key, down_key = jax.random.split(key)
        return DenseMLP(
            self,
            up_projection=self.linear_config.random_init_mixture(
                mixture_size,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
                key=up_key,
            ),
            down_projection=self.linear_config.random_init_mixture(
                mixture_size,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
                key=down_key,
            ),
        )

    def empty_mixture(
        self,
        mixture_size: int,
        model_dim: int,
        hidden_dim: int,
    ) -> "DenseMLP":
        return DenseMLP(
            self,
            up_projection=self.linear_config.empty_mixture(
                mixture_size,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
            ),
            down_projection=self.linear_config.empty_mixture(
                mixture_size,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
            ),
        )


class MLPBase[ConfigT: MLPConfig](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @abstractmethod
    def __call__(self, inputs: Float[Array, " channels"]) -> Float[Array, " channels"]: ...


class DenseMLP(MLPBase[DenseMLPConfig]):
    up_projection: LinearBase
    down_projection: LinearBase

    @property
    def activation_precision(self) -> DTypeLike:
        return self.up_projection.activation_precision

    @property
    def model_dim(self) -> int:
        return self.up_projection.input_dim

    @property
    def hidden_dim(self) -> int:
        return self.down_projection.input_dim

    def __post_init__(self) -> None:
        up_output_dim, gate_output_dim = self.up_projection.output_dims
        if up_output_dim != gate_output_dim:
            raise ValueError(
                f"Up projection output dimension {up_output_dim} does not match"
                f" the gate output dimension {gate_output_dim}",
            )
        (down_output_dim,) = self.down_projection.output_dims
        if self.up_projection.input_dim != down_output_dim:
            raise ValueError(
                f"Down projection input dimension {down_output_dim} does not match"
                f" the up projection output dimension {self.up_projection.input_dim}",
            )

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " channels"]) -> Float[Array, " channels"]:
        up_proj, gate = self.up_projection(inputs)
        if self.config.gate_clipping:
            gate = jnp.clip(gate, *self.config.gate_clipping)
        if self.config.up_clipping:
            up_proj = jnp.clip(up_proj, *self.config.up_clipping)
        gate = self.config.activation(gate)
        (result,) = self.down_projection(up_proj * gate)
        return result

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterTree:
        return {
            "up_projection": self.up_projection.export_weights(weight_layout),
            "down_projection": self.down_projection.export_weights(weight_layout),
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
        weight_layout: WeightLayout = WeightLayout.AUTO,
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["up_projection"], Mapping)
        assert isinstance(weights["down_projection"], Mapping)
        return replace(
            self,
            up_projection=self.up_projection.import_weights(weights["up_projection"], weight_layout),
            down_projection=self.down_projection.import_weights(weights["down_projection"], weight_layout),
        )


class RoutingMap(eqx.Module):
    expert_mask: Bool[Array, " experts"]
    expert_weights: Float[Array, " experts"]


@dataclass(frozen=True)
class RoutingFunctionBase(ABC):
    @abstractmethod
    def __call__(self, logits: Float[Array, " experts"], num_active: int) -> RoutingMap: ...


@dataclass(frozen=True)
class SoftmaxRouting(RoutingFunctionBase):
    def __call__(self, logits: Float[Array, " experts"], num_active: int) -> RoutingMap:
        active_logits, active_indices = jax.lax.top_k(logits, num_active)
        active_weights = jax.nn.softmax(active_logits)
        mask = jnp.zeros_like(logits, dtype=bool)
        mask = mask.at[active_indices].set(True)
        expert_weights = jnp.zeros_like(logits)
        expert_weights = expert_weights.at[active_indices].set(active_weights)
        return RoutingMap(expert_mask=mask, expert_weights=expert_weights)


RoutingFunction = SoftmaxRouting | DummyUnionMember


register_config_union(RoutingFunction)


@dataclass(frozen=True)
class MixtureOfExpertsConfig:
    mixture_size: int
    num_experts_per_token: int
    routing_function: RoutingFunctionBase

    router_config: LinearConfig
    router_has_biases: bool

    expert_config: DenseMLPConfig

    def random_init(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> "MixtureOfExperts":
        experts_key, router_key = jax.random.split(key)
        router = self.router_config.random_init(
            model_dim,
            (self.mixture_size,),
            has_biases=self.router_has_biases,
            key=router_key,
        )
        experts = self.expert_config.random_init_mixture(self.mixture_size, model_dim, hidden_dim, key=experts_key)
        return MixtureOfExperts(self, router, experts)

    def empty(self, model_dim: int, hidden_dim: int) -> "MixtureOfExperts":
        router = self.router_config.empty(model_dim, (self.mixture_size,), has_biases=self.router_has_biases)
        experts = self.expert_config.empty(model_dim, hidden_dim)
        return MixtureOfExperts(self, router, experts)


class MixtureOfExperts(MLPBase[MixtureOfExpertsConfig]):
    router: LinearBase
    experts: DenseMLP

    @property
    def mixture_size(self) -> int:
        return self.config.mixture_size

    @property
    def num_experts_per_token(self) -> int:
        return self.config.num_experts_per_token

    @property
    def activation_precision(self) -> DTypeLike:
        return self.experts.activation_precision

    @property
    def model_dim(self) -> int:
        return self.experts.model_dim

    @property
    def hidden_dim(self) -> int:
        return self.experts.hidden_dim

    def __call__(self, inputs: Float[Array, " channels"]) -> Float[Array, " channels"]:
        (router_logits,) = self.router(inputs)
        routing_map = self.config.routing_function(router_logits, self.num_experts_per_token)
        eqx

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterTree:
        return {
            "router": self.router.export_weights(weight_layout),
            "experts": self.experts.export_weights(weight_layout),
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
        weight_layout: WeightLayout = WeightLayout.AUTO,
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["router"], Mapping)
        assert isinstance(weights["experts"], Mapping)
        return replace(
            self,
            router=self.router.import_weights(weights["router"], weight_layout),
            experts=self.experts.import_weights(weights["experts"], weight_layout),
        )


MLPConfig = DenseMLPConfig | MixtureOfExpertsConfig


register_config_union(MLPConfig)
