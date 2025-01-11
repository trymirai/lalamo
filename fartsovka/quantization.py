from enum import Enum, auto

from jax import numpy as jnp
from jaxtyping import Array, Float

from .common import DType

__all__ = ["QuantizationMode", "quantize_weights"]


class QuantizationMode(Enum):
    INT4 = auto()
    INT8 = auto()

    @property
    def range(self) -> tuple[int, int]:
        return MODE_TO_RANGE[self]

    @property
    def dtype(self) -> DType:
        value_to_dtype = {
            QuantizationMode.INT4: jnp.int8,
            QuantizationMode.INT8: jnp.int8,
        }
        return value_to_dtype[self]


MODE_TO_RANGE = {
    QuantizationMode.INT4: (-8, 7),
    QuantizationMode.INT8: (-128, 127),
}


def quantize_weights(x: Float[Array, "..."], mode: QuantizationMode) -> Float[Array, "..."]:
    range_min, range_max = MODE_TO_RANGE[mode]
    return jnp.clip(jnp.round(x), range_min, range_max)


def dynamically_quantize_activations(
    x: Float[Array, " channels"],
    mode: QuantizationMode,
    eps: float = 1e07,
) -> Float[Array, " channels"]:
    # Find the maximum absolute value along the channels dimension
    max_value, min_value = jnp.max(x), jnp.min(x)
    zero_point = (max_value + min_value) / 2
    scale = (max_value - min_value + eps) / 2

    # Scale to [-1, 1] range
    x_normalized = (x - zero_point) / scale

    # Scale to target range and back to simulate quantization
    range_min, range_max = mode.range
    x_quantized = jnp.clip(x_normalized * range_max, range_min, range_max) / range_max

    # Scale back to original range
    return x_quantized * scale + zero_point
