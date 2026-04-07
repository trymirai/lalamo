from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from lalamo.modules.common import ShardingConfig

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Key

from .base import ArrayForwardPassConfig, CompressedArray, GradientEstimator
from .quantization_helpers import pack_uint_to_uint8, quantize_to_grid, unpack_uint8_to_uint


class MLXQuantArray(CompressedArray):
    weights: Float[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    biases: Float[Array, "... out_channels groups"]
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
        expanded_biases = jnp.repeat(self.biases, self.group_size, axis=-1)
        return quantized_weights * expanded_scales + expanded_biases

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        return self.dequantize(quantize_to_grid(self.weights, self.bits))

    @classmethod
    def compress(
        cls,
        weights: Float[Array, "... out_channels in_channels"],
        *,
        bits: int,
        group_size: int,
    ) -> "MLXQuantArray":
        grouped = rearrange(
            weights,
            "... out_channels (groups group_size) -> ... out_channels groups group_size",
            group_size=group_size,
        )
        group_mins = jnp.min(grouped, axis=-1)
        group_maxs = jnp.max(grouped, axis=-1)
        quant_levels = (2**bits) - 1

        scales = jnp.maximum((group_maxs - group_mins) / quant_levels, jnp.finfo(weights.dtype).eps)

        safe_scales = jnp.repeat(jnp.where(scales == 0, jnp.finfo(weights.dtype).eps, scales), group_size, axis=-1)
        expanded_biases = jnp.repeat(group_mins, group_size, axis=-1)
        float_weights = quantize_to_grid((weights - expanded_biases) / safe_scales, bits)

        return cls(weights=float_weights, scales=scales, biases=group_mins, bits=bits, group_size=group_size)

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
        return {
            "__class__": type(self).__name__,
            "bits": self.bits,
            "group_size": self.group_size,
            "weights": pack_uint_to_uint8(int_weights, self.bits),
            "scales": self.scales,
            "biases": self.biases,
        }

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: str = "",
        sharding_config: "ShardingConfig | None" = None,  # noqa: ARG002
    ) -> Self:
        def _key(name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        bits = int(data[_key("bits")])
        group_size = int(data[_key("group_size")])
        return type(self)(
            weights=unpack_uint8_to_uint(data[_key("weights")], bits).astype(data[_key("scales")].dtype),
            scales=data[_key("scales")],
            biases=data[_key("biases")],
            bits=bits,
            group_size=group_size,
        )
