from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterPath, ParameterTree, dummy_array

from .base import ArrayForwardPassConfig, CompressedArray, unpack_int32


def mlx_quantize_per_group(
    weights: Float[Array, "... out_channels in_channels"],
    scales: Float[Array, "... out_channels groups"],
    deq_biases: Float[Array, "... out_channels groups"],
    *,
    group_size: int,
    bits: int,
) -> Array:
    safe_scales = jnp.repeat(jnp.maximum(scales, jnp.finfo(weights.dtype).eps), group_size, axis=-1)
    expanded_biases = jnp.repeat(deq_biases, group_size, axis=-1)
    return jnp.clip(jnp.round((weights - expanded_biases) / safe_scales), 0, (2**bits) - 1)


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


class MLXQuantArray(CompressedArray):
    raw: Float[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    deq_biases: Float[Array, "... out_channels groups"]
    group_size: int = eqx.field(static=True)
    bits: int = eqx.field(static=True)

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

    def fake_quantize(self) -> Array:
        int_weights = mlx_quantize_per_group(
            self.raw, self.scales, self.deq_biases, group_size=self.group_size, bits=self.bits
        )
        return mlx_dequantize_per_group(int_weights, self.scales, self.deq_biases, group_size=self.group_size)

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:
        if not forward_pass_config.quantize:
            return self.raw
        quantized = self.fake_quantize()
        return self.raw + jax.lax.stop_gradient(quantized - self.raw)

    def export_weights(self) -> ParameterTree:
        int_dtype = jnp.uint4 if self.bits == 4 else jnp.uint8
        int_weights = mlx_quantize_per_group(
            self.raw, self.scales, self.deq_biases, group_size=self.group_size, bits=self.bits
        )
        return dict(
            weights=int_weights.astype(int_dtype),
            scales=self.scales,
            deq_biases=self.deq_biases,
        )

    @staticmethod
    def import_weights(
        weights_map: Mapping[str, Array],
        *,
        precision: DTypeLike,
        group_size: int,
        bits: int,
    ) -> MLXQuantArray:
        scales = weights_map["scales"]
        deq_biases = weights_map["deq_biases"]
        raw = mlx_dequantize_per_group(weights_map["weights"], scales, deq_biases, group_size=group_size)
        return MLXQuantArray(raw=raw, scales=scales, deq_biases=deq_biases, group_size=group_size, bits=bits)

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
        return MLXQuantArray(
            raw=dummy_array((*leading_dims, out_channels, in_channels), precision),
            scales=dummy_array((*leading_dims, out_channels, num_groups), precision),
            deq_biases=dummy_array((*leading_dims, out_channels, num_groups), precision),
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
        bits: int,
    ) -> MLXQuantArray:
        path = ParameterPath(prefix)
        unpacked_weights = unpack_int32(state_dict[path / "weight"], bits)
        scales = state_dict[path / "scales"].astype(dtype)
        deq_biases = state_dict[path / "biases"].astype(dtype)
        raw = mlx_dequantize_per_group(unpacked_weights.astype(dtype), scales, deq_biases, group_size=group_size)
        return MLXQuantArray(raw=raw, scales=scales, deq_biases=deq_biases, group_size=group_size, bits=bits)
