from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange

from lalamo.common import is_abstract_array

from .base import (
    ArrayForwardPassConfig,
    CompressedArray,
    DeterministicQuantize,
    NoQuantize,
    StochasticQuantize,
    pack_uint_to_uint8,
    unpack_int32,
    unpack_uint8_to_uint,
)

if TYPE_CHECKING:
    from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

    from lalamo.modules.common import Initializer

AWQ_UINT4_REVERSE_ORDER = jnp.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=jnp.int32)


def _reverse_awq_uint4_order(array: Array) -> Array:
    grouped = rearrange(array, "... (group pack) -> ... group pack", pack=8)
    return rearrange(grouped[..., AWQ_UINT4_REVERSE_ORDER], "... group pack -> ... (group pack)")


def awq_quantize_per_group(
    weights: Float[Array, "... out_channels in_channels"],
    scales: Float[Array, "... out_channels groups"],
    zero_points: Array,
    *,
    group_size: int,
    bits: int,
) -> Array:
    safe_scales = jnp.repeat(jnp.maximum(scales, jnp.finfo(weights.dtype).eps), group_size, axis=-1)
    expanded_zp = jnp.repeat(zero_points, group_size, axis=-1)
    return jnp.clip(jnp.round(weights / safe_scales + expanded_zp), 0, (2**bits) - 1)


def awq_stochastic_quantize_per_group(
    weights: Float[Array, "... out_channels in_channels"],
    scales: Float[Array, "... out_channels groups"],
    zero_points: Array,
    *,
    group_size: int,
    bits: int,
    key: PRNGKeyArray,
) -> Array:
    safe_scales = jnp.repeat(jnp.maximum(scales, jnp.finfo(weights.dtype).eps), group_size, axis=-1)
    expanded_zp = jnp.repeat(zero_points, group_size, axis=-1)
    pre_round = weights / safe_scales + expanded_zp
    return jnp.clip(jnp.floor(pre_round + jax.random.uniform(key, pre_round.shape)), 0, (2**bits) - 1)


def awq_dequantize_per_group(
    int_weights: Array,
    scales: Float[Array, "... out_channels groups"],
    zero_points: Array,
    *,
    group_size: int,
) -> Array:
    expanded_scales = jnp.repeat(scales, group_size, axis=-1)
    expanded_zp = jnp.repeat(zero_points, group_size, axis=-1)
    return (int_weights - expanded_zp) * expanded_scales


class AWQQuantArray(CompressedArray):
    raw: Float[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    zero_points: Array
    group_size: int = eqx.field(static=True)
    bits: int = eqx.field(static=True, default=4)

    @property
    def is_abstract(self) -> bool:
        return is_abstract_array(self.raw)

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:  # noqa: B008
        if self.is_abstract:
            return self.raw
        match forward_pass_config.quantize:
            case NoQuantize():
                return self.raw
            case DeterministicQuantize():
                int_weights = awq_quantize_per_group(
                    self.raw, self.scales, self.zero_points, group_size=self.group_size, bits=self.bits
                )
            case StochasticQuantize(key=key):
                int_weights = awq_stochastic_quantize_per_group(
                    self.raw, self.scales, self.zero_points, group_size=self.group_size, bits=self.bits, key=key
                )
            case unknown:
                raise TypeError(f"Unknown quantization strategy: {unknown}")
        quantized = awq_dequantize_per_group(int_weights, self.scales, self.zero_points, group_size=self.group_size)
        quantized = quantized.astype(self.raw.dtype)
        return self.raw + jax.lax.stop_gradient(quantized - self.raw)

    def to_uzu(self) -> dict[str, Array]:
        quantized_weights = awq_quantize_per_group(
            self.raw,
            self.scales,
            self.zero_points,
            group_size=self.group_size,
            bits=self.bits,
        )
        packed_weights = pack_uint_to_uint8(quantized_weights, self.bits)
        packed_zero_points = pack_uint_to_uint8(self.zero_points.astype(jnp.uint8), self.bits)
        return {
            "qweight": packed_weights,
            "scales": self.scales,
            "qzeros": packed_zero_points,
        }

    def from_uzu(self, weights: dict[str, Array]) -> AWQQuantArray:
        packed_weights = weights["qweight"]
        packed_zero_points = weights["qzeros"]
        float_dtype = weights["scales"].dtype
        unpacked_weights = unpack_uint8_to_uint(packed_weights, self.bits).astype(float_dtype)
        unpacked_zero_points = unpack_uint8_to_uint(packed_zero_points, self.bits).astype(float_dtype)
        raw = awq_dequantize_per_group(
            unpacked_weights,
            weights["scales"],
            unpacked_zero_points,
            group_size=self.group_size,
        )
        return type(self)(
            raw=raw,
            scales=weights["scales"],
            zero_points=unpacked_zero_points,
            group_size=self.group_size,
            bits=self.bits,
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
    ) -> AWQQuantArray:
        num_groups = in_channels // group_size
        return cls(
            raw=initializer.zeros((*leading_dims, out_channels, in_channels), initializer.precision),
            scales=initializer.ones((*leading_dims, out_channels, num_groups), initializer.precision),
            zero_points=initializer.zeros((*leading_dims, out_channels, num_groups), initializer.precision),
            group_size=group_size,
            bits=bits,
        )

    @classmethod
    def from_packed(
        cls,
        packed_weights: Array,
        packed_zero_points: Array,
        scales: Array,
        *,
        dtype: DTypeLike,
        group_size: int,
        bits: int,
    ) -> AWQQuantArray:
        unpacked_weights = unpack_int32(packed_weights, bits)
        unpacked_zero_points = unpack_int32(packed_zero_points, bits)
        if bits == 4:
            unpacked_weights = _reverse_awq_uint4_order(unpacked_weights)
            unpacked_zero_points = _reverse_awq_uint4_order(unpacked_zero_points)
        adjusted_scales = scales.T.astype(dtype)
        zero_points = unpacked_zero_points.T.astype(dtype)
        raw = awq_dequantize_per_group(
            unpacked_weights.T.astype(dtype),
            adjusted_scales,
            zero_points,
            group_size=group_size,
        )
        return cls(
            raw=raw,
            scales=adjusted_scales,
            zero_points=zero_points,
            group_size=group_size,
            bits=bits,
        )
