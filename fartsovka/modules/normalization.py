from dataclasses import dataclass

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar

from fartsovka.common import DEFAULT_PRECISION, DType

from .common import DummyUnionMember, FartsovkaModule, ParameterDict, register_config_union

__all__ = [
    "AbstractNormalization",
    "AbstractNormalizationConfig",
    "RMSNorm",
    "RMSNormConfig",
    "NormalizationConfigType",
]


class AbstractNormalization(FartsovkaModule):
    model_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        raise NotImplementedError


@dataclass
class AbstractNormalizationConfig[NormalizationType: AbstractNormalization]:
    def __call__(self, model_dim: int, eps: float) -> NormalizationType:
        raise NotImplementedError


def _compute_adjusted_variance(
    x: Float[Array, " channels"],
    eps: float,
    accumulation_precision: DType,
    result_precision: DType,
) -> Float[Scalar, ""]:
    upcasted_x = x.astype(accumulation_precision)
    result = jnp.mean(jnp.square(upcasted_x)) + eps
    return result.astype(result_precision)


class RMSNorm(AbstractNormalization):
    scale: Float[Array, " channels"]

    scale_precision: DType = eqx.field(static=True)
    accumulation_precision: DType = eqx.field(static=True)

    def __init__(
        self,
        *,
        model_dim: int,
        eps: float,
        scale_precision: DType,
        accumulation_precision: DType,
    ) -> None:
        super().__init__(model_dim=model_dim, eps=eps)

        self.accumulation_precision = accumulation_precision
        self.scale_precision = scale_precision
        self.scale = jnp.ones(model_dim, dtype=scale_precision)

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        adjusted_variance = _compute_adjusted_variance(x, self.eps, self.accumulation_precision, self.scale_precision)
        result = x * jax.lax.rsqrt(adjusted_variance) * self.scale
        return result.astype(x.dtype)

    def export_weights(self) -> ParameterDict:
        return ParameterDict(scale=self.scale)


@dataclass
class RMSNormConfig(AbstractNormalizationConfig[RMSNorm]):
    scale_precision: DType = DEFAULT_PRECISION
    accumulation_precision: DType = jnp.float32

    def __call__(self, model_dim: int, eps: float) -> RMSNorm:
        return RMSNorm(
            model_dim=model_dim,
            eps=eps,
            scale_precision=self.scale_precision,
            accumulation_precision=self.accumulation_precision,
        )


NormalizationConfigType = RMSNormConfig | DummyUnionMember

register_config_union(NormalizationConfigType)
