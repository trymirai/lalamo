from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

from lalamo.exportable import ExportResults
from lalamo.weight_matrix import Layout

__all__ = ["U4M3Z4QuantizedWeights"]


@dataclass(frozen=True)
class U4M3Z4QuantizedWeights:
    codes: Array
    output_scales: Array
    multipliers: Array
    zero_points: Array
    group_size: int = 32

    def __post_init__(self) -> None:
        if self.group_size not in (32, 64):
            raise ValueError("U4M3Z4 group_size must be 32 or 64")
        if self.codes.ndim != 2:
            raise ValueError("codes must have shape [rows, cols]")
        rows, cols = self.codes.shape
        if rows == 0 or cols == 0 or cols % (4 * self.group_size) != 0:
            raise ValueError("codes must have positive rows and columns, with columns divisible by 4 * group_size")
        groups = cols // self.group_size
        if self.output_scales.shape != (rows,):
            raise ValueError("output_scales must have shape [rows]")
        if self.multipliers.shape != (rows, groups) or self.zero_points.shape != (rows, groups):
            raise ValueError("multipliers and zero_points must have shape [rows, groups]")
        if self.codes.dtype != jnp.uint8 or self.multipliers.dtype != jnp.uint8 or self.zero_points.dtype != jnp.uint8:
            raise ValueError("codes, multipliers and zero_points must be uint8")
        if self.output_scales.dtype != jnp.float32:
            raise ValueError("output_scales must be float32")
        if bool(jnp.any(self.codes > 15)):
            raise ValueError("codes must be in 0..15")
        if bool(jnp.any((self.multipliers < 1) | (self.multipliers > 8))):
            raise ValueError("multipliers must be in 1..8")
        if bool(jnp.any(self.zero_points > 15)):
            raise ValueError("zero_points must be in 0..15")
        if bool(jnp.any(~jnp.isfinite(self.output_scales) | (self.output_scales <= 0))):
            raise ValueError("output_scales must be finite and positive")

    def export(self) -> ExportResults:
        rows = self.codes.shape[0]
        weights = self.codes[:, 0::2] | (self.codes[:, 1::2] << 4)
        packed_multipliers_and_zero_points = (self.multipliers - 1) | (self.zero_points << 3)
        packed_multipliers_and_zero_points = jnp.pad(
            packed_multipliers_and_zero_points.T,
            ((0, 0), (0, -rows % 4)),
        )
        return ExportResults(
            arrays={
                "weights": weights,
                "scales": self.output_scales,
                "multipliers_and_zero_points": packed_multipliers_and_zero_points,
            },
            metadata={
                "spec": {
                    "type": "U4M3Z4Spec",
                    "bits": 4,
                    "group_size": self.group_size,
                    "multiplier_bits": 3,
                    "zero_point_bits": 4,
                    "layout": Layout.OUTPUT_INPUT.value,
                },
            },
        )
