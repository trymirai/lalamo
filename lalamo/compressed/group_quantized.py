from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.common import ParameterPath
from lalamo.module import Initializer

from .base import ArrayForwardPassConfig, CompressedArray, CompressedArraySpec, GradientEstimator
from .quantization_helpers import pack_quant_weights, quantize_to_grid, unpack_quant_weights


@dataclass(frozen=True)
class AWQSpec(CompressedArraySpec):
    bits: int
    group_size: int
    float_dtype: DTypeLike = jnp.float32

    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> "AWQArray":
        grouped = rearrange(
            weights,
            "... out_channels (groups group_size) -> ... out_channels groups group_size",
            group_size=self.group_size,
        )
        group_mins = jnp.min(grouped, axis=-1)
        group_maxs = jnp.max(grouped, axis=-1)
        quant_levels = (2**self.bits) - 1

        scales = jnp.maximum((group_maxs - group_mins) / quant_levels, jnp.finfo(weights.dtype).eps)
        zero_points = jnp.clip(jnp.round(-group_mins / scales), 0, (2**self.bits) - 1)

        safe_scales = jnp.repeat(
            jnp.where(scales == 0, jnp.finfo(weights.dtype).eps, scales), self.group_size, axis=-1
        )
        expanded_zero_points = jnp.repeat(zero_points, self.group_size, axis=-1)
        float_weights = quantize_to_grid(weights / safe_scales + expanded_zero_points, self.bits)

        return AWQArray(spec=self, weights=float_weights, scales=scales, zero_points=zero_points)

    def init(
        self,
        initializer: "Initializer",
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> "AWQArray":
        num_groups = in_channels // self.group_size
        return AWQArray(
            spec=self,
            weights=initializer.zeros((*leading_dims, out_channels, in_channels), initializer.precision),
            scales=initializer.ones((*leading_dims, out_channels, num_groups), initializer.precision),
            zero_points=initializer.zeros((*leading_dims, out_channels, num_groups), initializer.precision),
        )

    def from_uzu(self, data: Mapping[str, Any], prefix: ParameterPath) -> "AWQArray":
        return AWQArray(
            spec=self,
            weights=unpack_quant_weights(data[prefix / "weights"], self.bits, self.float_dtype),
            scales=data[prefix / "scales"],
            zero_points=unpack_quant_weights(data[prefix / "zero_points"], self.bits, self.float_dtype),
        )


class AWQArray(CompressedArray[AWQSpec]):
    weights: Float[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    zero_points: Float[Array, "... out_channels groups"]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.scales.dtype

    def to_uzu(self) -> dict[str, Any]:
        result = super().to_uzu()
        result["weights"] = pack_quant_weights(self.weights, self.spec.bits)
        result["zero_points"] = pack_quant_weights(self.zero_points, self.spec.bits)
        return result

    def dequantize(
        self, quantized_weights: Float[Array, "... out_channels in_channels"]
    ) -> Float[Array, "... out_channels in_channels"]:
        expanded_scales = jnp.repeat(self.scales, self.spec.group_size, axis=-1)
        expanded_zero_points = jnp.repeat(self.zero_points, self.spec.group_size, axis=-1)
        return (quantized_weights - expanded_zero_points) * expanded_scales

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        return self.dequantize(quantize_to_grid(self.weights, self.spec.bits))

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: Key[Array, ""],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, "... out_channels"]:
        match forward_pass_config.gradient_estimator:
            case GradientEstimator.NONE:
                q = quantize_to_grid(self.weights, self.spec.bits)
            case GradientEstimator.DETERMINISTIC:
                q = quantize_to_grid(self.weights, self.spec.bits)
                q = self.weights + jax.lax.stop_gradient(q - self.weights)
            case GradientEstimator.STOCHASTIC_DROPOUT:
                q = quantize_to_grid(self.weights, self.spec.bits)
                q = self.weights + jax.lax.stop_gradient(q - self.weights)
                mean = self.dequantize(q) @ vector
                error_sq = jax.lax.stop_gradient((self.weights - q) ** 2)
                expanded_scales_sq = jnp.repeat(self.scales**2, self.spec.group_size, axis=-1)
                variance = (error_sq * expanded_scales_sq) @ (vector**2)
                return mean + jnp.sqrt(variance) * jax.random.normal(key, mean.shape, dtype=mean.dtype)
            case _:
                raise ValueError(f"Unhandled gradient estimator: {forward_pass_config.gradient_estimator}")
        return self.dequantize(q) @ vector
