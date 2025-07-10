from enum import Enum

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

__all__ = ["QuantizationMode", "quantize_weights"]


class QuantizationMode(Enum):
    UINT4 = "uint4"
    INT8 = "int8"
    UINT8 = "uint8"

    @classmethod
    def from_num_bits(cls, num_bits: int) -> "QuantizationMode":
        bit_to_mode = {
            4: cls.UINT4,
            8: cls.UINT8,
        }
        if num_bits not in bit_to_mode:
            raise ValueError(f"No quantization mode defined for {num_bits} bits")
        return bit_to_mode[num_bits]

    @property
    def range(self) -> tuple[int, int]:
        return MODE_TO_RANGE[self]

    @property
    def dtype(self) -> DTypeLike:
        value_to_dtype = {
            QuantizationMode.UINT4: jnp.uint4,
            QuantizationMode.INT8: jnp.int8,
            QuantizationMode.UINT8: jnp.uint8,
        }
        return value_to_dtype[self]

    @property
    def bits(self) -> int:
        value_to_bits = {
            QuantizationMode.UINT4: 4,
            QuantizationMode.INT8: 8,
            QuantizationMode.UINT8: 8,
        }
        return value_to_bits[self]

    def __str__(self) -> str:
        return self.value


MODE_TO_RANGE = {
    QuantizationMode.UINT4: (0, 15),
    QuantizationMode.INT8: (-128, 127),
    QuantizationMode.UINT8: (0, 255),
}


def quantize_weights(x: Float[Array, "..."], mode: QuantizationMode) -> Float[Array, "..."]:
    range_min, range_max = MODE_TO_RANGE[mode]
    return jnp.clip(jnp.round(x), range_min, range_max)


def dynamically_quantize_activations(
    x: Float[Array, " channels"],
    mode: QuantizationMode,
) -> Float[Array, " channels"]:
    # Reference implementation: https://github.com/pytorch/pytorch/blob/2ccbacfa24cae724ec1ea3bc7de189e5bf948d46/torch/ao/quantization/fx/_decomposed.py#L790
    range_min, range_max = mode.range
    min_val = jnp.min(x)
    max_val = jnp.max(x)
    min_val_neg = jnp.minimum(min_val, 0)
    max_val_pos = jnp.maximum(max_val, 0)

    # scale
    scale = (max_val_pos - min_val_neg) / (range_max - range_min)
    scale = jnp.maximum(scale, jnp.finfo(x.dtype).eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = range_min + descaled_min
    zero_point_from_max_error = range_max + descaled_max
    zero_point = jnp.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        range_min - descaled_min,
        range_max - descaled_max,
    )
    zero_point = jnp.round(jnp.clip(zero_point, range_min, range_max))

    x_normalized = x / scale + zero_point
    x_quantized = jnp.clip(jnp.round(x_normalized), range_min, range_max)

    return (x_quantized - zero_point) * scale
