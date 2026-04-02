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
        return Normalization(config=self, scales=scales, biases=biases)


class Normalization(LalamoModule[NormalizationConfig]):
    scales: Float[Array, " channels"]
    biases: Float[Array, " channels"] | None = None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.scale_precision

    @property
    def input_dim(self) -> int:
        (result,) = self.scales.shape
        return result

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
