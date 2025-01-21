from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .activations import silu
from .common import FartsovkaModule, ParameterDict
from .linear import LinearBase, LinearFactoryBase

__all__ = ["MLP", "MLPBase", "MLPFactory", "MLPFactoryBase"]


class MLPBase(FartsovkaModule):
    model_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        raise NotImplementedError


@dataclass
class MLPFactoryBase[MLPType: MLPBase]:
    def __call__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        use_bias: bool,
        key: PRNGKeyArray,
    ) -> MLPType:
        raise NotImplementedError


class MLP[LinearType: LinearBase](MLPBase):
    up_projection: LinearType
    down_projection: LinearType

    def __init__(
        self,
        linear_factory: LinearFactoryBase[LinearType],
        model_dim: int,
        hidden_dim: int,
        use_bias: bool,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(model_dim=model_dim, hidden_dim=hidden_dim, use_bias=use_bias)

        up_projection_key, down_projection_key = jax.random.split(key)
        self.up_projection = linear_factory(
            model_dim,
            (hidden_dim, hidden_dim),
            use_bias=use_bias,
            key=up_projection_key,
        )
        self.down_projection = linear_factory(
            hidden_dim,
            (model_dim,),
            use_bias=use_bias,
            key=down_projection_key,
        )

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        up_proj, gate = self.up_projection(x)
        gate = silu(gate)
        (result,) = self.down_projection(up_proj * gate)
        return result

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            up_projection=self.up_projection.export_weights(),
            down_projection=self.down_projection.export_weights(),
        )


@dataclass
class MLPFactory[LinearType: LinearBase](MLPFactoryBase[MLP[LinearType]]):
    linear_factory: LinearFactoryBase[LinearType]

    def __call__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        use_bias: bool,
        key: PRNGKeyArray,
    ) -> MLP[LinearType]:
        return MLP(
            linear_factory=self.linear_factory,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            use_bias=use_bias,
            key=key,
        )
