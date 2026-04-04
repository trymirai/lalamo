from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from lalamo.common import is_abstract_array
from lalamo.quantization import QuantizationMode

from .base import (
    ArrayForwardPassConfig,
    CompressedArray,
    DeterministicQuantize,
    NoQuantize,
    StochasticQuantize,
    unpack_int32,
)

if TYPE_CHECKING:
    from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray
    from lalamo.modules.common import Initializer


def mlx_quantize_per_group(
    weights: Float[Array, "... out_channels in_channels"],
    scales: Float[Array, "... out_channels groups"],
    deq_biases: Float[Array, "... out_channels groups"],
    *,
    group_size: int,
    bits: int,
) -> Array:
    safe_scales = jnp.repeat(jnp.where(scales == 0, jnp.finfo(weights.dtype).eps, scales), group_size, axis=-1)
    expanded_biases = jnp.repeat(deq_biases, group_size, axis=-1)
    return jnp.clip(jnp.round((weights - expanded_biases) / safe_scales), 0, (2**bits) - 1)


def mlx_stochastic_quantize_per_group(
    weights: Float[Array, "... out_channels in_channels"],
    scales: Float[Array, "... out_channels groups"],
    deq_biases: Float[Array, "... out_channels groups"],
    *,
    group_size: int,
    bits: int,
    key: PRNGKeyArray,
) -> Array:
    safe_scales = jnp.repeat(jnp.where(scales == 0, jnp.finfo(weights.dtype).eps, scales), group_size, axis=-1)
    expanded_biases = jnp.repeat(deq_biases, group_size, axis=-1)
    pre_round = (weights - expanded_biases) / safe_scales
    return jnp.clip(jnp.floor(pre_round + jax.random.uniform(key, pre_round.shape)), 0, (2**bits) - 1)


def mlx_dequantize_per_group(
    int_weights: Array,
    scales: Float[Array, "... out_channels groups"],
    deq_biases: Float[Array, "... out_channels groups"],
    *,
    group_size: int,
) -> Array:
    expanded_scales = jnp.repeat(scales, group_size, axis=-1)
    expanded_biases = jnp.repeat(deq_biases, group_size, axis=-1)
    return int_weights * expanded_scales + expanded_biases


def detect_mlx_quantization(
    expected_in_channels: int,
    packed_in_channels: int,
    default: QuantizationMode,
) -> QuantizationMode:
    if packed_in_channels * 8 == expected_in_channels:
        return QuantizationMode.UINT4
    if packed_in_channels * 4 == expected_in_channels:
        return QuantizationMode.UINT8
    if packed_in_channels * 32 == expected_in_channels:
        return QuantizationMode.UINT1
    return default


class MLXQuantArray(CompressedArray):
    raw: Float[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    deq_biases: Float[Array, "... out_channels groups"]
    group_size: int = eqx.field(static=True)
    bits: int = eqx.field(static=True)

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:  # noqa: B008
        if is_abstract_array(self.raw):
            return self.raw
        match forward_pass_config.quantize:
            case NoQuantize():
                return self.raw
            case DeterministicQuantize():
                int_weights = mlx_quantize_per_group(
                    self.raw, self.scales, self.deq_biases, group_size=self.group_size, bits=self.bits
                )
            case StochasticQuantize(key=key):
                int_weights = mlx_stochastic_quantize_per_group(
                    self.raw, self.scales, self.deq_biases, group_size=self.group_size, bits=self.bits, key=key
                )
        quantized = mlx_dequantize_per_group(int_weights, self.scales, self.deq_biases, group_size=self.group_size)
        return self.raw + jax.lax.stop_gradient(quantized - self.raw)

    def __check_init__(self) -> None:
        *_, out_ch, in_ch = self.raw.shape
        expected_groups = in_ch // self.group_size
        if in_ch % self.group_size != 0:
            raise ValueError(f"in_channels ({in_ch}) not divisible by group_size ({self.group_size})")
        *_, scale_out, scale_groups = self.scales.shape
        if scale_out != out_ch or scale_groups != expected_groups:
            raise ValueError(f"scales shape {self.scales.shape} incompatible with weights shape {self.raw.shape}")
        *_, bias_out, bias_groups = self.deq_biases.shape
        if bias_out != out_ch or bias_groups != expected_groups:
            raise ValueError(
                f"deq_biases shape {self.deq_biases.shape} incompatible with weights shape {self.raw.shape}"
            )

    @classmethod
    def init(
        cls,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
        *,
        group_size: int,
        bits: int,
    ) -> MLXQuantArray:
        num_groups = in_channels // group_size
        return cls(
            raw=initializer.zeros((*leading_dims, out_channels, in_channels), initializer.precision),
            scales=initializer.ones((*leading_dims, out_channels, num_groups), initializer.precision),
            deq_biases=initializer.zeros((*leading_dims, out_channels, num_groups), initializer.precision),
            group_size=group_size,
            bits=bits,
        )

    @classmethod
    def from_packed(
        cls,
        packed_weights: Array,
        scales: Array,
        deq_biases: Array,
        *,
        dtype: DTypeLike,
        expected_in_channels: int,
        group_size: int,
        bits: int,
    ) -> MLXQuantArray:
        quantization_mode = detect_mlx_quantization(
            expected_in_channels,
            packed_weights.shape[-1],
            QuantizationMode.from_num_bits(bits),
        )
        unpacked_weights = unpack_int32(packed_weights, quantization_mode.bits).astype(dtype)
        adjusted_scales = scales.astype(dtype)
        adjusted_biases = deq_biases.astype(dtype)
        raw = mlx_dequantize_per_group(
            unpacked_weights,
            adjusted_scales,
            adjusted_biases,
            group_size=group_size,
        )
        return cls(
            raw=raw,
            scales=adjusted_scales,
            deq_biases=adjusted_biases,
            group_size=group_size,
            bits=quantization_mode.bits,
        )
