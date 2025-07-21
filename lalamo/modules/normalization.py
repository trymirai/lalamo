from dataclasses import dataclass
from enum import Enum

import jax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterDict

from .common import LalamoModule, WeightLayout

__all__ = [
    "RMSNorm",
    "RMSNormConfig",
    "UpcastMode",
]


class UpcastMode(Enum):
    ONLY_NORMALIZATION = "only_normalization"
    FULL_LAYER = "full_layer"


@dataclass(frozen=True)
class RMSNormConfig:
    scale_precision: DTypeLike
    accumulation_precision: DTypeLike
    epsilon: float
    scale_offset: float | None
    upcast_mode: UpcastMode

    def init(self, channels: int) -> "RMSNorm":
        scales = jnp.ones(channels, dtype=self.scale_precision)
        return RMSNorm(self, scales=scales)


class RMSNorm(LalamoModule[RMSNormConfig]):
    scales: Float[Array, " channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.scale_precision

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

    def __call__(self, inputs: Float[Array, " channels"]) -> Float[Array, " channels"]:
        upcasted_inputs = inputs.astype(self.config.accumulation_precision)

        adjusted_variance = jnp.mean(jnp.square(upcasted_inputs)) + self.config.epsilon
        normalized_x = upcasted_inputs * jax.lax.rsqrt(adjusted_variance)

        if self.config.upcast_mode == UpcastMode.ONLY_NORMALIZATION:
            normalized_x = normalized_x.astype(inputs.dtype)

        if self.config.upcast_mode == UpcastMode.FULL_LAYER:
            adjusted_scales = self.scales.astype(self.config.accumulation_precision)
        else:
            adjusted_scales = self.scales

        if self.config.scale_offset is not None:
            adjusted_scales = adjusted_scales + self.config.scale_offset

        result = normalized_x * adjusted_scales
        return result.astype(inputs.dtype)

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:  # noqa: ARG002
        return ParameterDict(scales=self.scales)
