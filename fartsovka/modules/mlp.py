from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .activations import Activation
from .common import DummyUnionMember, FartsovkaModule, ParameterDict, register_config_union
from .linear import AbstractLinear, AbstractLinearConfig, LinearConfigType

__all__ = ["MLP", "AbstractMLP", "MLPConfig", "AbstractMLPConfig", "MLPConfigType"]


class AbstractMLP(FartsovkaModule):
    model_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    activation: Activation = eqx.field(static=True)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        raise NotImplementedError


@dataclass
class AbstractMLPConfig[MLPType: AbstractMLP]:
    def __call__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        use_bias: bool,
        activation: Activation,
        key: PRNGKeyArray,
    ) -> MLPType:
        raise NotImplementedError


class MLP[LinearType: AbstractLinear](AbstractMLP):
    up_projection: LinearType
    down_projection: LinearType

    def __init__(
        self,
        linear_config: AbstractLinearConfig[LinearType],
        model_dim: int,
        hidden_dim: int,
        use_bias: bool,
        activation: Activation,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(model_dim=model_dim, hidden_dim=hidden_dim, use_bias=use_bias, activation=activation)

        up_projection_key, down_projection_key = jax.random.split(key)
        self.up_projection = linear_config(
            model_dim,
            (hidden_dim, hidden_dim),
            use_bias=use_bias,
            key=up_projection_key,
        )
        self.down_projection = linear_config(
            hidden_dim,
            (model_dim,),
            use_bias=use_bias,
            key=down_projection_key,
        )

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        up_proj, gate = self.up_projection(x)
        gate = self.activation(gate)
        (result,) = self.down_projection(up_proj * gate)
        return result

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            up_projection=self.up_projection.export_weights(),
            down_projection=self.down_projection.export_weights(),
        )


@dataclass
class MLPConfig[LinearType: AbstractLinear](AbstractMLPConfig[MLP[LinearType]]):
    linear_config: LinearConfigType

    def __call__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        use_bias: bool,
        activation: Activation,
        key: PRNGKeyArray,
    ) -> MLP[LinearType]:
        return MLP(
            linear_config=self.linear_config,  # type: ignore
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            use_bias=use_bias,
            activation=activation,
            key=key,
        )


MLPConfigType = MLPConfig | DummyUnionMember


register_config_union(MLPConfigType)
