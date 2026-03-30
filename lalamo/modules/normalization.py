from dataclasses import dataclass
from enum import Enum

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree, require_mapping

from .common import Initializer, LalamoModule

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
    use_bias: bool = False

    def init(self, initializer: Initializer, input_dim: int) -> "Normalization":
        scales = initializer.ones((input_dim,), self.scale_precision)
        if self.use_bias:
            biases = initializer.zeros((input_dim,), self.scale_precision)
        else:
            biases = None
        return Normalization(
            scales=scales,
            biases=biases,
            activation_precision=self.scale_precision,
            accumulation_precision=self.accumulation_precision,
            epsilon=self.epsilon,
            scale_offset=self.scale_offset,
            upcast_mode=self.upcast_mode,
            subtract_mean=self.subtract_mean,
        )


class Normalization(LalamoModule):
    scales: Float[Array, " channels"]

    activation_precision: DTypeLike = eqx.field(static=True)
    accumulation_precision: DTypeLike = eqx.field(static=True)

    epsilon: float = eqx.field(static=True)
    scale_offset: float | None = eqx.field(static=True)
    upcast_mode: UpcastMode = eqx.field(static=True)
    subtract_mean: bool = eqx.field(static=True)

    biases: Float[Array, " channels"] | None = None

    @property
    def input_dim(self) -> int:
        (result,) = self.scales.shape
        return result

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " channels"]) -> Float[Array, " channels"]:
        upcasted_inputs = inputs.astype(self.accumulation_precision)

        if self.subtract_mean:
            mean = jnp.mean(upcasted_inputs)
            upcasted_inputs = upcasted_inputs - mean

        adjusted_variance = jnp.mean(jnp.square(upcasted_inputs)) + self.epsilon
        normalized_x = upcasted_inputs * jax.lax.rsqrt(adjusted_variance)

        if self.upcast_mode == UpcastMode.ONLY_NORMALIZATION:
            normalized_x = normalized_x.astype(inputs.dtype)

        if self.upcast_mode == UpcastMode.FULL_LAYER:
            adjusted_scales = self.scales.astype(self.accumulation_precision)
        else:
            adjusted_scales = self.scales

        if self.scale_offset is not None:
            adjusted_scales = adjusted_scales + self.scale_offset

        result = normalized_x * adjusted_scales

        if self.biases is not None:
            result += self.biases
        return result.astype(inputs.dtype)

    def export_weights(self) -> ParameterTree:
        result: dict[str, Array] = {"scales": self.scales}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        weights = require_mapping(weights)
        if self.biases is not None:
            assert isinstance(weights["biases"], Array)
            biases = weights["biases"]
        else:
            biases = None
        return replace(self, scales=weights["scales"], biases=biases)
