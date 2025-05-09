from dataclasses import dataclass

import jax
from jax import numpy as jnp, Array
from jaxtyping import Array, Float, Scalar, PRNGKeyArray

from fartsovka.common import DType, ParameterDict
from fartsovka.samples import ModuleSample, DecoderSamplesContext

from .common import FartsovkaModule

__all__ = [
    "RMSNorm",
    "RMSNormConfig",
]


def _inverse_rms(
    x: Float[Array, " channels"],
    eps: float,
    accumulation_precision: DType,
    result_precision: DType,
) -> Float[Scalar, ""]:
    upcasted_x = x.astype(accumulation_precision)
    result = jnp.mean(jnp.square(upcasted_x)) + eps
    return result.astype(result_precision)


@dataclass
class RMSNormConfig:
    scale_precision: DType
    accumulation_precision: DType
    epsilon: float

    def init(self, channels: int) -> "RMSNorm":
        scales = jnp.ones(channels, dtype=self.scale_precision)
        return RMSNorm(self, scales=scales)


class RMSNorm(FartsovkaModule[RMSNormConfig]):
    scales: Float[Array, " channels"]

    @property
    def input_dim(self) -> int:
        (result,) = self.scales.shape
        return result

    def __post_init__(self) -> None:
        if self.config.scale_precision != self.scales.dtype:
            raise ValueError(
                f"Scales precision {self.scales.dtype} does not match the"
                f" specified precision {self.config.scale_precision}",
            )

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        adjusted_variance = _inverse_rms(
            x,
            self.config.epsilon,
            self.config.accumulation_precision,
            self.scales.dtype,
        )
        result = x * jax.lax.rsqrt(adjusted_variance) * self.scales
        return result.astype(x.dtype)

    def export_weights(self) -> ParameterDict:
        return ParameterDict(scales=self.scales)

    def export_samples(self, context: DecoderSamplesContext, key: PRNGKeyArray) -> ParameterDict:
        x = jax.random.uniform(
            key,
            (context.suffix_length, self.input_dim),
            minval=-1,
            maxval=1,
            dtype=self.config.scale_precision,
        )
        y = self(x)
        sample = ModuleSample(inputs=(x,), outputs=(y,))
        return ParameterDict(value=sample.export())
