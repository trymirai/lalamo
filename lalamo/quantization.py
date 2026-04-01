from enum import Enum
from functools import partial

import jax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Key

__all__ = ["QuantizationMode", "quantize_weights", "stochastic_quantize_weights"]


class QuantizationMode(Enum):
    UINT1 = "uint1"
    UINT4 = "uint4"
    UINT8 = "uint8"

    @classmethod
    def from_num_bits(cls, num_bits: int) -> "QuantizationMode":
        match num_bits:
            case 1:
                return cls.UINT1
            case 4:
                return cls.UINT4
            case 8:
                return cls.UINT8
            case _:
                raise ValueError(f"No quantization mode defined for {num_bits} bits")

    @property
    def range(self) -> tuple[int, int]:
        match self:
            case QuantizationMode.UINT1:
                return (0, 1)
            case QuantizationMode.UINT4:
                return (0, 15)
            case QuantizationMode.UINT8:
                return (0, 255)

    @property
    def dtype(self) -> DTypeLike:
        match self:
            case QuantizationMode.UINT1:
                return jnp.uint8
            case QuantizationMode.UINT4:
                return jnp.uint4
            case QuantizationMode.UINT8:
                return jnp.uint8

    @property
    def bits(self) -> int:
        match self:
            case QuantizationMode.UINT1:
                return 1
            case QuantizationMode.UINT4:
                return 4
            case QuantizationMode.UINT8:
                return 8

    def __str__(self) -> str:
        return self.value


def _quantize_weights_primal(x: Float[Array, "..."], mode: QuantizationMode) -> Float[Array, "..."]:
    range_min, range_max = mode.range
    return jnp.clip(jnp.round(x), range_min, range_max)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def quantize_weights(x: Float[Array, "..."], mode: QuantizationMode) -> Float[Array, "..."]:
    return _quantize_weights_primal(x, mode)


def _quantize_weights_fwd(
    x: Float[Array, "..."],
    mode: QuantizationMode,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    return _quantize_weights_primal(x, mode), x


def _quantize_weights_bwd(
    mode: QuantizationMode,
    residuals: Float[Array, "..."],
    grad_output: Float[Array, "..."],
) -> tuple[Float[Array, "..."]]:
    x = residuals
    range_min, range_max = mode.range
    gradient_mask = jnp.logical_and(x >= range_min, x <= range_max)
    return (grad_output * gradient_mask.astype(grad_output.dtype),)


quantize_weights.defvjp(_quantize_weights_fwd, _quantize_weights_bwd)


def stochastic_quantize_weights(
    x: Float[Array, "..."],
    mode: QuantizationMode,
    key: Key[Array, ""],
) -> Float[Array, "..."]:
    range_min, range_max = mode.range
    clipped = jnp.clip(x, range_min, range_max)
    lower = jnp.floor(clipped)
    upper = jnp.ceil(clipped)
    upper_probability = clipped - lower
    samples = jax.random.uniform(key, clipped.shape, dtype=jnp.float32)
    return jnp.where(samples < upper_probability, upper, lower)


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
