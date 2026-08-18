from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule

__all__ = [
    "Normalization",
    "NormalizationConfig",
    "UpcastMode",
]


class UpcastMode(StrEnum):
    ONLY_NORMALIZATION = "only_normalization"
    FULL_LAYER = "full_layer"


@dataclass(frozen=True)
class NormalizationConfig(LalamoConfig):
    epsilon: float
    scale_offset: float | None
    upcast_mode: UpcastMode
    subtract_mean: bool
    has_scale: bool = True
    has_biases: bool = False

    def __post_init__(self) -> None:
        if not self.has_scale and self.scale_offset is not None:
            raise ValueError("scale_offset is meaningless for a weightless normalization.")

    def init(self, initializer: Initializer, input_dim: int) -> "Normalization":
        if self.has_scale:
            scales = initializer.ones((input_dim,), dtype=jnp.float32)
        else:
            scales = None
        if self.has_biases:
            biases = initializer.zeros((input_dim,), dtype=jnp.float32)
        else:
            biases = None
        return Normalization(
            config=self,
            sharding_config=initializer.sharding_config,
            scales=scales,
            biases=biases,
        )


class Normalization(LalamoModule[NormalizationConfig]):
    scales: Float[Array, " channels"] | None
    biases: Float[Array, " channels"] | None

    @property
    def input_dim(self) -> int:
        assert self.config.has_scale and self.scales is not None, "weightless normalization does not track input_dim"
        (result,) = self.scales.shape
        return result

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, " channels"],
        accumulation_precision: DTypeLike = jnp.float32,
    ) -> Float[Array, " channels"]:
        upcasted_inputs = inputs.astype(accumulation_precision)

        if not self.config.has_scale:
            scales = None
        else:
            assert self.scales is not None, "has_scale is set but scales are missing"
            if self.config.upcast_mode == UpcastMode.FULL_LAYER:
                scales = self.scales.astype(jnp.float32)
            else:
                scales = self.scales.astype(inputs.dtype)

        if scales is not None and self.config.scale_offset is not None:
            scales += self.config.scale_offset

        if self.config.subtract_mean:
            mean = jnp.mean(upcasted_inputs)
            upcasted_inputs = upcasted_inputs - mean

        adjusted_variance = jnp.mean(jnp.square(upcasted_inputs)) + self.config.epsilon
        normalized_x = upcasted_inputs * jax.lax.rsqrt(adjusted_variance)

        if self.config.upcast_mode == UpcastMode.ONLY_NORMALIZATION:
            normalized_x = normalized_x.astype(inputs.dtype)

        if scales is not None:
            result = normalized_x * scales
        else:
            result = normalized_x

        if self.biases is not None:
            result = result + self.biases.astype(result.dtype)

        return result.astype(inputs.dtype)
