from dataclasses import dataclass
from enum import Enum

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from fartsovka.common import DType, ParameterDict

from .common import FartsovkaModule, WeightLayout

__all__ = [
    "RMSNorm",
    "RMSNormConfig",
    "UpcastMode",
]


class UpcastMode(Enum):
    ONLY_NORMALIZATION = "only_normalization"
    FULL_LAYER = "full_layer"


@dataclass
class RMSNormConfig:
    scale_precision: DType
    accumulation_precision: DType
    epsilon: float
    scale_offset: float | None
    upcast_mode: UpcastMode

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
        x = x.astype(self.config.accumulation_precision)

        adjusted_variance = jnp.mean(jnp.square(x)) + self.config.epsilon
        normalized_x = x * jax.lax.rsqrt(adjusted_variance)

        if self.config.upcast_mode == UpcastMode.ONLY_NORMALIZATION:
            normalized_x = normalized_x.astype(x.dtype)

        if self.config.upcast_mode == UpcastMode.FULL_LAYER:
            adjusted_scales = self.scales.astype(self.config.accumulation_precision)
        else:
            adjusted_scales = self.scales

        if self.config.scale_offset is not None:
            adjusted_scales = adjusted_scales + self.config.scale_offset

        result = normalized_x * adjusted_scales
        return result.astype(x.dtype)

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.INPUT_OUTPUT) -> ParameterDict:  # noqa: ARG002
        return ParameterDict(scales=self.scales)
