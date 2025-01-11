from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .activations import silu
from .linear import LinearBase, LinearFactoryBase

__all__ = ["MLPBase", "MLPFactoryBase", "MLP", "MLPFactory"]


class MLPBase(eqx.Module):
    model_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        raise NotImplementedError


@dataclass
class MLPFactoryBase[MLPType: MLPBase]:
    def __call__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray) -> MLPType:
        raise NotImplementedError


class MLP[LinearType: LinearBase](MLPBase):
    up_projection: LinearType
    down_projection: LinearType

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        linear_factory: LinearFactoryBase[LinearType],
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(model_dim=model_dim, hidden_dim=hidden_dim)

        up_projection_key, down_projection_key = jax.random.split(key)
        self.up_projection = linear_factory(
            model_dim,
            (hidden_dim, hidden_dim),
            key=up_projection_key,
        )
        self.down_projection = linear_factory(
            hidden_dim,
            (model_dim,),
            key=down_projection_key,
        )

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        up_proj, gate = self.up_projection(x)
        gate = silu(gate)
        (result,) = self.down_projection(up_proj * gate)
        return result


@dataclass
class MLPFactory[LinearType: LinearBase](MLPFactoryBase[MLP[LinearType]]):
    linear_factory: LinearFactoryBase[LinearType]

    def __call__(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> MLP[LinearType]:
        return MLP(
            model_dim,
            hidden_dim,
            linear_factory=self.linear_factory,
            key=key,
        )
