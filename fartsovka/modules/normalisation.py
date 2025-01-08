from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float

from .common import DEFAULT_PRECISION

__all__ = ["NormalisationBase", "NormalisationFactory"]


class NormalisationBase(eqx.Module):
    model_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        raise NotImplementedError


@dataclass
class NormalisationFactory[NormalisationType: NormalisationBase]:
    def __call__(self, model_dim: int, eps: float) -> NormalisationType:
        raise NotImplementedError


def _compute_adjusted_variance(
    x: Float[Array, " channels"],
    eps: float,
    accumulation_precision: jnp.dtype,
) -> Float[Array, " channels"]:
    upcasted_x = x.astype(accumulation_precision)
    result = jnp.mean(upcasted_x**2, axis=-1, keepdims=True) + eps
    return result.astype(x.dtype)


class RMSNorm(NormalisationBase):
    scale: Float[Array, " channels"]

    precision: jnp.dtype = eqx.field(static=True, default=DEFAULT_PRECISION)
    accumulation_precision: jnp.dtype = eqx.field(static=True, default=jnp.float32)

    def __init__(self, model_dim: int, eps: float, precision: jnp.dtype, accumulation_precision: jnp.dtype) -> None:
        self.model_dim = model_dim
        self.eps = eps
        self.precision = precision
        self.accumulation_precision = accumulation_precision
        self.scale = jnp.ones(model_dim, dtype=accumulation_precision)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        adjusted_variance = _compute_adjusted_variance(x, self.eps, self.precision)
        return x * self.scale * (1 / jnp.sqrt(adjusted_variance))


@dataclass
class RMSNormFactory(NormalisationFactory[RMSNorm]):
    precision: jnp.dtype = dataclass_field(default=DEFAULT_PRECISION)
    accumulation_precision: jnp.dtype = dataclass_field(default=jnp.float32)

    def __call__(self, model_dim: int, eps: float) -> RMSNorm:
        return RMSNorm(model_dim, eps, self.precision, self.accumulation_precision)
