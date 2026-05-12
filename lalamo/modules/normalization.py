from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
import tokamax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.utils.sharding import make_sharding, with_sharding

__all__ = [
    "Normalization",
    "NormalizationConfig",
    "NormalizationForwardPassConfig",
    "NormalizationImplementation",
    "UpcastMode",
]


class UpcastMode(StrEnum):
    ONLY_NORMALIZATION = "only_normalization"
    FULL_LAYER = "full_layer"


class NormalizationImplementation(StrEnum):
    STANDARD = "standard"
    TOKAMAX = "tokamax"


@dataclass(frozen=True)
class NormalizationForwardPassConfig:
    implementation: NormalizationImplementation = NormalizationImplementation.STANDARD
    accumulation_precision: DTypeLike | None = jnp.float32

    @classmethod
    def for_tracer_tests(cls) -> "NormalizationForwardPassConfig":
        return cls(implementation=NormalizationImplementation.STANDARD)

    @classmethod
    def for_inference(cls) -> "NormalizationForwardPassConfig":
        return cls(implementation=NormalizationImplementation.TOKAMAX)

    @classmethod
    def for_training(cls) -> "NormalizationForwardPassConfig":
        return cls(implementation=NormalizationImplementation.STANDARD)


@dataclass(frozen=True)
class NormalizationConfig(LalamoConfig):
    epsilon: float
    scale_offset: float | None
    upcast_mode: UpcastMode
    subtract_mean: bool
    has_biases: bool = False

    def init(self, initializer: Initializer, input_dim: int) -> "Normalization":
        scales = initializer.ones((input_dim,))
        if self.has_biases:
            biases = initializer.zeros((input_dim,))
        else:
            biases = None
        return Normalization(config=self, scales=scales, biases=biases)


class Normalization(LalamoModule[NormalizationConfig]):
    scales: Float[Array, " channels"]
    biases: Float[Array, " channels"] | None

    @property
    def input_dim(self) -> int:
        (result,) = self.scales.shape
        return result

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, " channels"],
        forward_pass_config: NormalizationForwardPassConfig = NormalizationForwardPassConfig(),
        accumulation_precision: DTypeLike | None = None,
    ) -> Float[Array, " channels"]:
        accumulation_precision = accumulation_precision or forward_pass_config.accumulation_precision or inputs.dtype
        upcasted_inputs = with_sharding(inputs.astype(accumulation_precision), make_sharding((None,)))

        scale_dtype = accumulation_precision if self.config.upcast_mode == UpcastMode.FULL_LAYER else inputs.dtype
        scales = with_sharding(self.scales.astype(scale_dtype), make_sharding((None,)))
        biases = None
        if self.biases is not None:
            biases = with_sharding(self.biases.astype(scale_dtype), make_sharding((None,)))

        match forward_pass_config.implementation:
            case NormalizationImplementation.STANDARD:
                if self.config.scale_offset is not None:
                    scales = scales + self.config.scale_offset

                if self.config.subtract_mean:
                    mean = jnp.mean(upcasted_inputs)
                    upcasted_inputs = upcasted_inputs - mean

                adjusted_variance = jnp.mean(jnp.square(upcasted_inputs)) + self.config.epsilon
                normalized_x = upcasted_inputs * jax.lax.rsqrt(adjusted_variance)

                if self.config.upcast_mode == UpcastMode.ONLY_NORMALIZATION:
                    normalized_x = normalized_x.astype(inputs.dtype)

                result = normalized_x * scales

                if biases is not None:
                    result += biases
                return with_sharding(result.astype(inputs.dtype), make_sharding((None,)))

            case NormalizationImplementation.TOKAMAX:
                result = tokamax.layer_norm(
                    upcasted_inputs,
                    scale=scales,
                    offset=biases,
                    epsilon=self.config.epsilon,
                    scale_offset=self.config.scale_offset if self.config.scale_offset is not None else 0.0,
                    subtract_mean=self.config.subtract_mean,
                ).astype(inputs.dtype)
                return with_sharding(result, make_sharding((None,)))
