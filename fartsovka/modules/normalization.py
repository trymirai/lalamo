from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar

from fartsovka.common import DEFAULT_PRECISION, DType

__all__ = ["NormalizationBase", "NormalizationFactoryBase", "RMSNorm", "RMSNormFactory"]


class NormalizationBase(eqx.Module):
    model_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        raise NotImplementedError


@dataclass
class NormalizationFactoryBase[NormalizationType: NormalizationBase]:
    def __call__(self, model_dim: int, eps: float) -> NormalizationType:
        raise NotImplementedError


def _compute_adjusted_variance(
    x: Float[Array, " channels"],
    eps: float,
    accumulation_precision: DType,
) -> Float[Scalar, ""]:
    upcasted_x = x.astype(accumulation_precision)
    result = jnp.mean(upcasted_x**2) + eps
    return result.astype(x.dtype)


class RMSNorm(NormalizationBase):
    scale: Float[Array, " channels"]

    precision: DType = eqx.field(static=True, default=DEFAULT_PRECISION)
    accumulation_precision: DType = eqx.field(static=True, default=jnp.float32)

    def __init__(
        self,
        model_dim: int,
        eps: float,
        precision: DType,
        accumulation_precision: DType,
    ) -> None:
        self.model_dim = model_dim
        self.eps = eps
        self.precision = precision
        self.accumulation_precision = accumulation_precision
        self.scale = jnp.ones(model_dim, dtype=accumulation_precision)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        adjusted_variance = _compute_adjusted_variance(x, self.eps, self.precision)
        return x * self.scale * (1 / jnp.sqrt(adjusted_variance))


@dataclass
class RMSNormFactory(NormalizationFactoryBase[RMSNorm]):
    precision: DType = dataclass_field(default=DEFAULT_PRECISION)
    accumulation_precision: DType = dataclass_field(default=jnp.float32)

    def __call__(self, model_dim: int, eps: float) -> RMSNorm:
        return RMSNorm(model_dim, eps, self.precision, self.accumulation_precision)
