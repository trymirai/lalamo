from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from lalamo.modules.common import ShardingConfig

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.common import ParameterPath

from .base import ArrayForwardPassConfig, CompressedArray, GradientEstimator
from .quantization_helpers import pack_uint_to_uint8, quantize_to_grid, unpack_uint8_to_uint


class AWQQuantArray(CompressedArray):
    weights: Float[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    zero_points: Float[Array, "... out_channels groups"]
    bits: int = eqx.field(static=True)
    group_size: int = eqx.field(static=True)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.scales.dtype

    def dequantize(
        self, quantized_weights: Float[Array, "... out_channels in_channels"]
    ) -> Float[Array, "... out_channels in_channels"]:
        expanded_scales = jnp.repeat(self.scales, self.group_size, axis=-1)
        expanded_zero_points = jnp.repeat(self.zero_points, self.group_size, axis=-1)
        return (quantized_weights - expanded_zero_points) * expanded_scales

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        return self.dequantize(quantize_to_grid(self.weights, self.bits))

    @classmethod
    def compress(
        cls,
        weights: Float[Array, "... out_channels in_channels"],
        *,
        bits: int,
        group_size: int,
    ) -> "AWQQuantArray":
        grouped = rearrange(
            weights,
            "... out_channels (groups group_size) -> ... out_channels groups group_size",
            group_size=group_size,
        )
        group_mins = jnp.min(grouped, axis=-1)
        group_maxs = jnp.max(grouped, axis=-1)
        quant_levels = (2**bits) - 1

        scales = jnp.maximum((group_maxs - group_mins) / quant_levels, jnp.finfo(weights.dtype).eps)
        zero_points = jnp.clip(jnp.round(-group_mins / scales), 0, (2**bits) - 1)

        safe_scales = jnp.repeat(jnp.where(scales == 0, jnp.finfo(weights.dtype).eps, scales), group_size, axis=-1)
        expanded_zero_points = jnp.repeat(zero_points, group_size, axis=-1)
        float_weights = quantize_to_grid(weights / safe_scales + expanded_zero_points, bits)

        return cls(weights=float_weights, scales=scales, zero_points=zero_points, bits=bits, group_size=group_size)

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: Key[Array, ""] | None,
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, "... out_channels"]:
        match forward_pass_config.gradient_estimator:
            case GradientEstimator.NONE:
                q = quantize_to_grid(self.weights, self.bits)
            case GradientEstimator.DETERMINISTIC:
                q = quantize_to_grid(self.weights, self.bits)
                q = self.weights + jax.lax.stop_gradient(q - self.weights)
            case GradientEstimator.STOCHASTIC_DROPOUT:
                assert key is not None
                q = quantize_to_grid(self.weights, self.bits)
                q = self.weights + jax.lax.stop_gradient(q - self.weights)
                mean = self.dequantize(q) @ vector
                error_sq = jax.lax.stop_gradient((self.weights - q) ** 2)
                expanded_scales_sq = jnp.repeat(self.scales**2, self.group_size, axis=-1)
                variance = (error_sq * expanded_scales_sq) @ (vector**2)
                return mean + jnp.sqrt(variance) * jax.random.normal(key, mean.shape, dtype=mean.dtype)
            case _:
                raise ValueError(f"Unhandled gradient estimator: {forward_pass_config.gradient_estimator}")
        return self.dequantize(q) @ vector

    def to_uzu(self) -> dict[str, Any]:
        int_weights = quantize_to_grid(self.weights, self.bits).astype(jnp.uint8)
        int_zero_points = quantize_to_grid(self.zero_points, self.bits).astype(jnp.uint8)
        return {
            "__class__": type(self).__name__,
            "bits": self.bits,
            "group_size": self.group_size,
            "weights": pack_uint_to_uint8(int_weights, self.bits),
            "scales": self.scales,
            "zero_points": pack_uint_to_uint8(int_zero_points, self.bits),
        }

    @classmethod
    def from_uzu(
        cls,
        data: Mapping[str, Any],
        prefix: str = "",
        sharding_config: "ShardingConfig | None" = None,  # noqa: ARG003
    ) -> Self:
        key = ParameterPath(prefix)
        bits = int(data[key / "bits"])
        group_size = int(data[key / "group_size"])
        return cls(
            weights=unpack_uint8_to_uint(data[key / "weights"], bits).astype(data[key / "scales"].dtype),
            scales=data[key / "scales"],
            zero_points=unpack_uint8_to_uint(data[key / "zero_points"], bits).astype(data[key / "scales"].dtype),
            bits=bits,
            group_size=group_size,
        )
