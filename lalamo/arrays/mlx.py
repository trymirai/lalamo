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


class MLXQuantArray(CompressedArray):
    int_weights: Float[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    deq_biases: Float[Array, "... out_channels groups"]
    group_size: int = eqx.field(static=True)
    bits: int = eqx.field(static=True)

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
        *_, bias_out, bias_groups = self.deq_biases.shape
        if bias_out != out_ch or bias_groups != expected_groups:
            raise ValueError(
                f"deq_biases shape {self.deq_biases.shape} incompatible with weights shape {self.int_weights.shape}"
            )

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.int_weights.shape, self.scales.dtype)

    def dot(self, vector: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        return self.value @ vector

    @property
    def value(self) -> Array:
        num_groups = self.int_weights.shape[-1] // self.group_size
        grouped_weights = rearrange(self.int_weights, "... o (g c) -> ... o g c", g=num_groups)
        per_group_scale = rearrange(self.scales, "... o g -> ... o g 1")
        per_group_bias = rearrange(self.deq_biases, "... o g -> ... o g 1")
        dequantized = grouped_weights * per_group_scale + per_group_bias
        return rearrange(dequantized, "... o g c -> ... o (g c)")

    def export_weights(self) -> ParameterTree:
        if self.bits == 4:
            packed_weights = jax_uint4_to_packed_uint8(self.int_weights.astype(jnp.uint4))
        else:
            packed_weights = self.int_weights.astype(jnp.uint8)
        return dict(weights=packed_weights, scales=self.scales, deq_biases=self.deq_biases)

    @staticmethod
    def empty(
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
        precision: DTypeLike,
        *,
        group_size: int,
        bits: int,
    ) -> MLXQuantArray:
        num_groups = in_channels // group_size
        zeros = lambda shape: dummy_array(shape, precision)
        return MLXQuantArray(
            int_weights=zeros((*leading_dims, out_channels, in_channels)),
            scales=zeros((*leading_dims, out_channels, num_groups)),
            deq_biases=zeros((*leading_dims, out_channels, num_groups)),
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
    ) -> MLXQuantArray:
        raw_weights = weights_map["weights"]
        if bits == 4:
            raw_weights = jax_uint8_to_unpacked_uint4(raw_weights)
        return MLXQuantArray(
            int_weights=raw_weights.astype(precision),
            scales=weights_map["scales"],
            deq_biases=weights_map["deq_biases"],
            group_size=group_size,
            bits=bits,
        )

    @staticmethod
    def from_weights(
        int_weights: Array,
        scales: Array,
        deq_biases: Array,
        *,
        group_size: int,
        bits: int,
    ) -> MLXQuantArray:
        return MLXQuantArray(
            int_weights=int_weights,
            scales=scales,
            deq_biases=deq_biases,
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
        bits: int,
    ) -> MLXQuantArray:
        unpacked_weights = unpack_int32(state_dict[f"{prefix}.weight"], bits)
        return MLXQuantArray(
            int_weights=unpacked_weights.astype(dtype),
            scales=state_dict[f"{prefix}.scales"].astype(dtype),
            deq_biases=state_dict[f"{prefix}.biases"].astype(dtype),
            group_size=group_size,
            bits=bits,
        )
