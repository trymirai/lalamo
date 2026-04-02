from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.core
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree, dummy_array
from lalamo.utils import jax_uint4_to_packed_uint8, jax_uint8_to_unpacked_uint4

from .base import CompressedArray, unpack_int32

AWQ_UINT4_REVERSE_ORDER = jnp.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=jnp.int32)


def _reverse_awq_uint4_order(array: Array) -> Array:
    grouped = rearrange(array, "... (group pack) -> ... group pack", pack=8)
    return rearrange(grouped[..., AWQ_UINT4_REVERSE_ORDER], "... group pack -> ... (group pack)")


class AWQQuantArray(CompressedArray):
    int_weights: Float[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    zero_points: Float[Array, "... out_channels groups"]
    group_size: int = eqx.field(static=True)
    bits: int = eqx.field(static=True, default=4)

    def __check_init__(self) -> None:
        *_, out_ch, in_ch = self.int_weights.shape
        expected_groups = in_ch // self.group_size
        if in_ch % self.group_size != 0:
            raise ValueError(f"in_channels ({in_ch}) not divisible by group_size ({self.group_size})")
        *_, scale_out, scale_groups = self.scales.shape
        if scale_out != out_ch or scale_groups != expected_groups:
            raise ValueError(
                f"scales shape {self.scales.shape} incompatible with weights shape {self.int_weights.shape}"
            )
        *_, zp_out, zp_groups = self.zero_points.shape
        if zp_out != out_ch or zp_groups != expected_groups:
            raise ValueError(
                f"zero_points shape {self.zero_points.shape} incompatible with weights shape {self.int_weights.shape}"
            )

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.int_weights.shape, self.scales.dtype)

    def dot(self, vector: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        return self.value @ vector

    @property
    def value(self) -> Array:
        num_groups = self.int_weights.shape[-1] // self.group_size
        grouped_weights = rearrange(self.int_weights, "... o (g c) -> ... o g c", g=num_groups)
        per_group_zp = rearrange(self.zero_points, "... o g -> ... o g 1")
        per_group_scale = rearrange(self.scales, "... o g -> ... o g 1")
        dequantized = (grouped_weights - per_group_zp) * per_group_scale
        return rearrange(dequantized, "... o g c -> ... o (g c)")

    def export_weights(self) -> ParameterTree:
        if self.bits == 4:
            packed_weights = jax_uint4_to_packed_uint8(self.int_weights.astype(jnp.uint4))
            packed_zero_points = jax_uint4_to_packed_uint8(self.zero_points.astype(jnp.uint4))
        else:
            packed_weights = self.int_weights.astype(jnp.uint8)
            packed_zero_points = self.zero_points.astype(jnp.uint8)
        return dict(weights=packed_weights, scales=self.scales, zero_points=packed_zero_points)

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
        zeros = lambda shape: dummy_array(shape, precision)
        return AWQQuantArray(
            int_weights=zeros((*leading_dims, out_channels, in_channels)),
            scales=zeros((*leading_dims, out_channels, num_groups)),
            zero_points=zeros((*leading_dims, out_channels, num_groups)),
            group_size=group_size,
            bits=bits,
        )

    @staticmethod
    def import_weights(
        weights_map: Mapping[str, Array],
        *,
        precision: DTypeLike,
        group_size: int,
        bits: int,
    ) -> AWQQuantArray:
        raw_weights = weights_map["weights"]
        raw_zero_points = weights_map["zero_points"]
        if bits == 4:
            raw_weights = jax_uint8_to_unpacked_uint4(raw_weights)
            raw_zero_points = jax_uint8_to_unpacked_uint4(raw_zero_points)
        return AWQQuantArray(
            int_weights=raw_weights.astype(precision),
            scales=weights_map["scales"],
            zero_points=raw_zero_points.astype(precision),
            group_size=group_size,
            bits=bits,
        )

    @staticmethod
    def from_weights(
        int_weights: Array,
        scales: Array,
        zero_points: Array,
        *,
        group_size: int,
        bits: int = 4,
    ) -> AWQQuantArray:
        return AWQQuantArray(
            int_weights=int_weights,
            scales=scales,
            zero_points=zero_points,
            group_size=group_size,
            bits=bits,
        )

    @staticmethod
    def from_torch(
        state_dict: Mapping[str, Array],
        *,
        prefix: str,
        dtype: DTypeLike,
        group_size: int,
        bits: int = 4,
    ) -> AWQQuantArray:
        unpacked_weights = unpack_int32(state_dict[f"{prefix}.qweight"], bits)
        unpacked_zeros = unpack_int32(state_dict[f"{prefix}.qzeros"], bits)

        if bits == 4:
            unpacked_weights = _reverse_awq_uint4_order(unpacked_weights)
            unpacked_zeros = _reverse_awq_uint4_order(unpacked_zeros)

        return AWQQuantArray(
            int_weights=unpacked_weights.T.astype(dtype),
            scales=state_dict[f"{prefix}.scales"].T.astype(dtype),
            zero_points=unpacked_zeros.T.astype(dtype),
            group_size=group_size,
            bits=bits,
        )
