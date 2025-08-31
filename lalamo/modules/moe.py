from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.mlp import MLP, MLPConfig

from .common import DummyUnionMember, LalamoModule, WeightLayout, register_config_union

# __all__ = ["MOE", "MOEConfig"]


class RoutingMap(eqx.Module):
    expert_ids: Float[Array, " active_experts"]
    expert_weights: Float[Array, " active_experts"]


@dataclass(frozen=True)
class RoutingFunctionBase(ABC):
    @abstractmethod
    def __call__(self, logits: Float[Array, " experts"], num_active: int) -> RoutingMap: ...


@dataclass(frozen=True)
class SoftmaxRouting(RoutingFunctionBase):
    def __call__(self, logits: Float[Array, " experts"], num_active: int) -> RoutingMap:
        active_logits, active_indices = jax.lax.top_k(logits, num_active)
        expert_weights = jax.nn.softmax(active_logits)
        return RoutingMap(expert_ids=active_indices, expert_weights=expert_weights)


RoutingFunction = register_config_union(SoftmaxRouting | DummyUnionMember)


@dataclass(frozen=True)
class MixtureOfExpertsConfig:
    num_experts_per_token: int
    routing_function: RoutingFunctionBase
    router_config: LinearConfig
    mlp_config: MLPConfig

    def random_init(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> "MixtureOfExperts":
        raise NotImplementedError

    def empty(self, model_dim: int, hidden_dim: int) -> "MixtureOfExperts":
        raise NotImplementedError


class MixtureOfExperts(LalamoModule[MixtureOfExpertsConfig]):
    router: LinearBase
    experts: MLP

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
