from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray

from lalamo.modules.common import Initializer
from lalamo.modules.forward_pass_config import ArrayForwardPassConfig

from .base import (
    CompressedArray,
    pack_uint_to_uint8,
    unpack_uint8_to_uint,
)


class AWQQuantArray(CompressedArray, kind="awq"):
    weights: Int[Array, "... out_channels in_channels"]
    scales: Float[Array, "... out_channels groups"]
    zero_points: Int[Array, "... out_channels groups"]
    bits: int
    group_size: int

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        expanded_scales = jnp.repeat(self.scales, self.group_size, axis=-1)
        expanded_zero_points = jnp.repeat(self.zero_points, self.group_size, axis=-1)
        return (
            self.weights.astype(self.scales.dtype) - expanded_zero_points.astype(self.scales.dtype)
        ) * expanded_scales

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
        zero_points = jnp.round(-group_mins / scales).astype(jnp.int32)

        safe_scales = jnp.repeat(jnp.where(scales == 0, jnp.finfo(weights.dtype).eps, scales), group_size, axis=-1)
        expanded_zero_points = jnp.repeat(zero_points.astype(weights.dtype), group_size, axis=-1)
        int_weights = jnp.clip(jnp.round(weights / safe_scales + expanded_zero_points), 0, quant_levels).astype(
            jnp.int32
        )

        return cls(weights=int_weights, scales=scales, zero_points=zero_points, bits=bits, group_size=group_size)

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: PRNGKeyArray | None,  # noqa: ARG002
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: ARG002, B008
    ) -> Float[Array, "... out_channels"]:
        return self.materialize() @ vector

    def to_uzu(self) -> dict[str, Any]:
        return {
            "__kind__": self.kind,
            "bits": self.bits,
            "group_size": self.group_size,
            "weights": pack_uint_to_uint8(self.weights.astype(jnp.uint8), self.bits),
            "scales": self.scales,
            "zero_points": pack_uint_to_uint8(self.zero_points.astype(jnp.uint8), self.bits),
        }

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> "AWQQuantArray":
        bits = int(data["bits"])
        group_size = int(data["group_size"])
        return cls(
            weights=unpack_uint8_to_uint(data["weights"], bits).astype(jnp.int32),
            scales=data["scales"],
            zero_points=unpack_uint8_to_uint(data["zero_points"], bits).astype(jnp.int32),
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
    ) -> "AWQQuantArray":
        num_groups = in_channels // group_size
        return cls(
            weights=initializer.zeros((*leading_dims, out_channels, in_channels), jnp.int32),
            scales=initializer.ones((*leading_dims, out_channels, num_groups), initializer.precision),
            zero_points=initializer.zeros((*leading_dims, out_channels, num_groups), jnp.int32),
            bits=bits,
            group_size=group_size,
        )
