"""Trained Dash weights as MLX matrices, without refitting.

A Dash weight is an odd code c in [-top, top] times a scale s. MLX stores q * scale + bias with q in [0, 2^bits),
so q = (c + top) / 2, scale = 2s and bias = -top * s. This is exact when the scales are float32; in bfloat16 the
bias rounds.
Scales are written per 128 columns, the group size of uzu's 2-bit path: a per-row scale is repeated across its row.
"""

from enum import Enum
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.utils.sharding import ShardingConfig, with_sharding

from .hybrid import HybridMatrix
from .incoherence import HadamardProcessor
from .mlx import MLXMatrix, MLXSpec
from .utils.packing import pack_uint_to_uint8

__all__ = ["DashCodes", "dash_matrix", "dash_mlx_matrix"]


class DashCodes(Enum):
    TWO_BIT = 3  # body and head: codes {-3, -1, 1, 3}
    THREE_BIT = 7  # input embedding: codes {-7, ..., 7}; MLX has no 3-bit mode, so it uses the 4-bit container

    @property
    def mlx_bits(self) -> Literal[2, 4]:
        match self:
            case DashCodes.TWO_BIT:
                return 2
            case DashCodes.THREE_BIT:
                return 4


def dash_mlx_matrix(
    codes: Int[Array, "rows columns"],
    scales: Float[Array, "rows groups"],
    *,
    code_range: DashCodes,
    sharding_config: ShardingConfig,
) -> MLXMatrix:
    group_size = 128
    rows, columns = codes.shape
    top = code_range.value
    assert columns % group_size == 0, columns
    assert scales.shape[0] == rows and (columns // group_size) % scales.shape[1] == 0, (codes.shape, scales.shape)
    assert bool(jnp.all((codes % 2 != 0) & (jnp.abs(codes) <= top))), f"codes must be odd and within +-{top}"
    spec = MLXSpec(bits=code_range.mlx_bits, group_size=group_size)
    weight_sharding = sharding_config.resolve_sharding(spec.layout.weight_partition(0, is_sharded=True))
    group_scales = jnp.repeat(scales, columns // group_size // scales.shape[1], axis=1)
    levels = with_sharding(((codes.astype(jnp.int32) + top) // 2).astype(jnp.uint8), weight_sharding)
    return spec.from_packed_parameters(
        packed_weights=pack_uint_to_uint8(levels, code_range.mlx_bits, sharding_config=sharding_config),
        scales=2 * group_scales,
        biases=-top * group_scales,
        sharding_config=sharding_config,
    )


def dash_matrix(
    codes: Int[Array, "rows columns"],
    scales: Float[Array, "rows groups"],
    input_signs: Int[Array, " columns"],
    *,
    sharding_config: ShardingConfig,
) -> HybridMatrix:
    """A Dash body or head linear: 2-bit codes in the Hadamard-128 basis of the signed input, no output rotation."""
    quantized = dash_mlx_matrix(codes, scales, code_range=DashCodes.TWO_BIT, sharding_config=sharding_config)
    return HybridMatrix.of(quantized, HadamardProcessor.of(input_signs, None, block_size=128))
