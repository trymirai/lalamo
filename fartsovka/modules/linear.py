import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .common import DEFAULT_PRECISION, DType

__all__ = ["LinearBase", "LinearFactoryBase", "Linear", "LinearFactory"]


class LinearBase(eqx.Module):
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __call__(self, x: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        raise NotImplementedError


@dataclass
class LinearFactoryBase[LinearType: LinearBase]:
    def __call__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray) -> LinearType:
        raise NotImplementedError


class Linear(LinearBase):
    weights: Float[Array, "out_channels in_channels"]

    precision: DType = eqx.field(static=True, default=DEFAULT_PRECISION)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        precision: DType = DEFAULT_PRECISION,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.precision = precision
        max_abs_value = 1 / math.sqrt(input_dim)
        self.weights = jax.random.uniform(
            key,
            (output_dim, input_dim),
            minval=-max_abs_value,
            maxval=max_abs_value,
            dtype=self.precision,
        )

    def __call__(self, x: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        return self.weights @ x


@dataclass
class LinearFactory(LinearFactoryBase[Linear]):
    precision: DType = dataclass_field(default=DEFAULT_PRECISION)

    def __call__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray) -> Linear:
        return Linear(input_dim, output_dim, precision=self.precision, key=key)
