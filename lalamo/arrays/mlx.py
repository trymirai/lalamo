from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.modules.common import Initializer

from .base import ArrayForwardPassConfig, CompressedArray, GradientEstimator
from .quantization_helpers import (
    pack_uint_to_uint8,
    quantize_to_grid,
    stochastic_quantize_to_grid,
    unpack_uint8_to_uint,
)


class MLXQuantArray(CompressedArray, kind="mlx"):
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
        key: PRNGKeyArray | None,
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, "... out_channels"]:
        match forward_pass_config.gradient_estimator:
            case GradientEstimator.NONE:
                q = quantize_to_grid(self.weights, self.bits)
            case GradientEstimator.DETERMINISTIC:
                q = quantize_to_grid(self.weights, self.bits)
                q = self.weights + jax.lax.stop_gradient(q - self.weights)
            case GradientEstimator.STOCHASTIC:
                assert key is not None
                q = stochastic_quantize_to_grid(self.weights, self.bits, key)
                q = self.weights + jax.lax.stop_gradient(q - self.weights)
            case _:
                raise ValueError(f"Unhandled gradient estimator: {forward_pass_config.gradient_estimator}")
        return self.dequantize(q) @ vector

    def to_uzu(self) -> dict[str, Any]:
        int_weights = quantize_to_grid(self.weights, self.bits).astype(jnp.uint8)
        return {
            "__kind__": self.kind,
            "bits": self.bits,
            "group_size": self.group_size,
            "weights": pack_uint_to_uint8(int_weights, self.bits),
            "scales": self.scales,
            "biases": self.biases,
        }

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> CompressedArray:
        if str(data.get("__kind__")) != cls.kind:
            return CompressedArray.from_uzu(data)
        bits = int(data["bits"])
        group_size = int(data["group_size"])
        return cls(
            weights=unpack_uint8_to_uint(data["weights"], bits).astype(data["scales"].dtype),
            scales=data["scales"],
            biases=data["biases"],
            bits=bits,
            group_size=group_size,
        )

    @classmethod
    def init(
        cls,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
        *,
        bits: int,
        group_size: int,
    ) -> "MLXQuantArray":
        num_groups = in_channels // group_size
        return cls(
            weights=initializer.zeros((*leading_dims, out_channels, in_channels), initializer.precision),
            scales=initializer.ones((*leading_dims, out_channels, num_groups), initializer.precision),
            biases=initializer.zeros((*leading_dims, out_channels, num_groups), initializer.precision),
            bits=bits,
            group_size=group_size,
        )
