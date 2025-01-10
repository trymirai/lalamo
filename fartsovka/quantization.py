from enum import Enum, auto

from jax import numpy as jnp
from jaxtyping import Array, Float

from .common import DType

__all__ = ["QuantizationMode", "quantize"]


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
    QuantizationMode.INT4: (-15, 15),
    QuantizationMode.INT8: (-127, 127),
}


def quantize(x: Float[Array, "..."], mode: QuantizationMode) -> Float[Array, "..."]:
    range_min, range_max = MODE_TO_RANGE[mode]
    return jnp.clip(jnp.round(x), range_min, range_max)
