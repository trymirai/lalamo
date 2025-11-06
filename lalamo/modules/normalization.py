from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Self

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree, dummy_array

from .common import LalamoModule

__all__ = [
    "Normalization",
    "NormalizationConfig",
    "UpcastMode",
]


class UpcastMode(Enum):
    ONLY_NORMALIZATION = "only_normalization"
    FULL_LAYER = "full_layer"


@dataclass(frozen=True)
class NormalizationConfig:
    scale_precision: DTypeLike
    accumulation_precision: DTypeLike
    epsilon: float
    scale_offset: float | None
    upcast_mode: UpcastMode
    subtract_mean: bool

    def init(self, input_dim: int) -> "Normalization":
        scales = jnp.ones(input_dim, dtype=self.scale_precision)
        return Normalization(self, scales=scales)

    def empty(self, input_dim: int) -> "Normalization":
        return Normalization(
            config=self,
            scales=dummy_array(input_dim, dtype=self.scale_precision),
        )


class Normalization(LalamoModule[NormalizationConfig]):
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

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " channels"]) -> Float[Array, " channels"]:
        upcasted_inputs = inputs.astype(self.config.accumulation_precision)

        if self.config.subtract_mean:
            mean = jnp.mean(upcasted_inputs)
            upcasted_inputs = upcasted_inputs - mean

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

    def export_weights(self) -> ParameterTree:
        return {"scales": self.scales}

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        return replace(self, scales=weights["scales"])
