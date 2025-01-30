from dataclasses import dataclass

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar

from fartsovka.common import DEFAULT_PRECISION, DType

from .common import FartsovkaModule, ModuleConfig, ParameterDict

__all__ = ["AbstractNormalization", "AbstractNormalizationConfig", "RMSNorm", "RMSNormConfig"]


class AbstractNormalization(FartsovkaModule):
    model_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        raise NotImplementedError


@dataclass
class AbstractNormalizationConfig[NormalizationType: AbstractNormalization](ModuleConfig[NormalizationType]):
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


class RMSNorm(AbstractNormalization):
    scale: Float[Array, " channels"]

    precision: DType = eqx.field(static=True)
    accumulation_precision: DType = eqx.field(static=True)

    def __init__(
        self,
        model_dim: int,
        eps: float,
        precision: DType,
        accumulation_precision: DType,
    ) -> None:
        super().__init__(model_dim=model_dim, eps=eps)

        self.precision = precision
        self.accumulation_precision = accumulation_precision
        self.scale = jnp.ones(model_dim, dtype=precision)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        adjusted_variance = _compute_adjusted_variance(x, self.eps, self.accumulation_precision)
        return x * jax.lax.rsqrt(adjusted_variance) * self.scale

    def export_weights(self) -> ParameterDict:
        return ParameterDict(scale=self.scale)


@dataclass
class RMSNormConfig(AbstractNormalizationConfig[RMSNorm]):
    precision: DType = DEFAULT_PRECISION
    accumulation_precision: DType = jnp.float32

    def __call__(self, model_dim: int, eps: float) -> RMSNorm:
        return RMSNorm(model_dim, eps, self.precision, self.accumulation_precision)
