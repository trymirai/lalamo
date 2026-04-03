from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterPath, ParameterTree, dummy_array

from .base import ArrayForwardPassConfig, CompressedArray, unpack_int32

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

    def __check_init__(self) -> None:
        *_, out_ch, in_ch = self.raw.shape
        expected_groups = in_ch // self.group_size
        if in_ch % self.group_size != 0:
            raise ValueError(f"in_channels ({in_ch}) not divisible by group_size ({self.group_size})")
        *_, scale_out, scale_groups = self.scales.shape
        if scale_out != out_ch or scale_groups != expected_groups:
            raise ValueError(f"scales shape {self.scales.shape} incompatible with weights shape {self.raw.shape}")
        *_, zp_out, zp_groups = self.zero_points.shape
        if zp_out != out_ch or zp_groups != expected_groups:
            raise ValueError(
                f"zero_points shape {self.zero_points.shape} incompatible with weights shape {self.raw.shape}"
            )

    def fake_quantize(self) -> Array:
        int_weights = awq_quantize_per_group(
            self.raw, self.scales, self.zero_points, group_size=self.group_size, bits=self.bits
        )
        return awq_dequantize_per_group(int_weights, self.scales, self.zero_points, group_size=self.group_size)

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:
        if not forward_pass_config.quantize:
            return self.raw
        quantized = self.fake_quantize()
        return self.raw + jax.lax.stop_gradient(quantized - self.raw)

    def export_weights(self) -> ParameterTree:
        int_dtype = jnp.uint4 if self.bits == 4 else jnp.uint8
        int_weights = awq_quantize_per_group(
            self.raw, self.scales, self.zero_points, group_size=self.group_size, bits=self.bits
        )
        return dict(
            weights=int_weights.astype(int_dtype),
            scales=self.scales,
            zero_points=self.zero_points.astype(int_dtype),
        )

    @staticmethod
    def import_weights(
        weights_map: Mapping[str, Array],
        *,
        precision: DTypeLike,
        group_size: int,
        bits: int,
    ) -> AWQQuantArray:
        scales = weights_map["scales"]
        zero_points = weights_map["zero_points"]
        raw = awq_dequantize_per_group(weights_map["weights"], scales, zero_points, group_size=group_size)
        return AWQQuantArray(raw=raw, scales=scales, zero_points=zero_points, group_size=group_size, bits=bits)

    @staticmethod
    def empty(
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
        precision: DTypeLike,
        *,
        group_size: int,
        bits: int,
    ) -> AWQQuantArray:
        num_groups = in_channels // group_size
        return AWQQuantArray(
            raw=dummy_array((*leading_dims, out_channels, in_channels), precision),
            scales=dummy_array((*leading_dims, out_channels, num_groups), precision),
            zero_points=jnp.zeros((*leading_dims, out_channels, num_groups), dtype=precision),
            group_size=group_size,
            bits=bits,
        )

    @staticmethod
    def from_torch(
        state_dict: Mapping[str, Array],
        *,
        prefix: ParameterPath | str,
        dtype: DTypeLike,
        group_size: int,
        bits: int = 4,
    ) -> AWQQuantArray:
        path = ParameterPath(prefix)
        unpacked_weights = unpack_int32(state_dict[path / "qweight"], bits)
        unpacked_zeros = unpack_int32(state_dict[path / "qzeros"], bits)
        if bits == 4:
            unpacked_weights = _reverse_awq_uint4_order(unpacked_weights)
            unpacked_zeros = _reverse_awq_uint4_order(unpacked_zeros)
        zero_points = unpacked_zeros.T.astype(dtype)
        scales = state_dict[path / "scales"].T.astype(dtype)
        raw = awq_dequantize_per_group(unpacked_weights.T.astype(dtype), scales, zero_points, group_size=group_size)
        return AWQQuantArray(raw=raw, scales=scales, zero_points=zero_points, group_size=group_size, bits=bits)
