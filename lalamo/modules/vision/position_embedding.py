import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import numpy as np
from einops import einsum
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule

from .image_grid import ImageGrid

__all__ = [
    "InterpolatedPositionEmbedding",
    "InterpolatedPositionEmbeddingConfig",
]


class BilinearTaps(NamedTuple):
    """Four table rows per patch and their blend weights, ordered top-left, top-right, bottom-left, bottom-right."""

    indices: Int[np.ndarray, "corners patches"]
    weights: Float[np.ndarray, "corners patches"]


@dataclass(frozen=True)
class InterpolatedPositionEmbeddingConfig(LalamoConfig):
    """Learned absolute position embedding, stretched to any image size by bilinear interpolation.

    The table is trained on a fixed ``side x side`` grid (48 x 48 for Qwen3.5). An image with an
    ``h x w`` patch grid samples that table at ``h`` evenly spaced rows and ``w`` evenly spaced
    columns, blending the four nearest entries.
    """

    num_positions: int

    def __post_init__(self) -> None:
        if math.isqrt(self.num_positions) ** 2 != self.num_positions:
            raise ValueError(f"num_positions must be a perfect square, got {self.num_positions}.")

    @property
    def side(self) -> int:
        return math.isqrt(self.num_positions)

    def bilinear_taps(self, grid: ImageGrid, spatial_merge_size: int) -> BilinearTaps:
        """Interpolation taps for one frame, in encoder (merge-block) patch order.

        These depend only on the static grid, so they are computed with NumPy at trace time and
        become constants in the compiled program. At run time only the gather and blend remain.
        The arithmetic mirrors HF's ``fast_pos_embed_interpolate``: float32 ``linspace``,
        truncation towards zero for the lower neighbour, and the upper neighbour clamped to the
        table edge.
        """
        last = self.side - 1

        def axis_taps(length: int) -> tuple[Int[np.ndarray, " n"], Int[np.ndarray, " n"], Float[np.ndarray, " n"]]:
            positions = np.linspace(0, last, length, dtype=np.float32)
            lower = positions.astype(np.int64)
            upper = np.minimum(lower + 1, last)
            return lower, upper, positions - lower.astype(np.float32)

        row_lower, row_upper, row_fraction = axis_taps(grid.height_in_patches)
        col_lower, col_upper, col_fraction = axis_taps(grid.width_in_patches)

        rows, cols = grid.patch_coordinates(spatial_merge_size)
        top, bottom, down = row_lower[rows], row_upper[rows], row_fraction[rows]
        left, right, across = col_lower[cols], col_upper[cols], col_fraction[cols]
        up, back = 1 - down, 1 - across

        return BilinearTaps(
            indices=np.stack(
                [
                    top * self.side + left,
                    top * self.side + right,
                    bottom * self.side + left,
                    bottom * self.side + right,
                ],
            ),
            weights=np.stack([up * back, up * across, down * back, down * across]),
        )

    def init(self, initializer: Initializer, model_dim: int) -> "InterpolatedPositionEmbedding":
        return InterpolatedPositionEmbedding(
            config=self,
            sharding_config=initializer.sharding_config,
            table=initializer.normal(0.02, (self.num_positions, model_dim)),
        )


class InterpolatedPositionEmbedding(LalamoModule[InterpolatedPositionEmbeddingConfig]):
    table: Float[Array, "positions channels"]

    @property
    def model_dim(self) -> int:
        _, result = self.table.shape
        return result

    @eqx.filter_jit
    def __call__(self, grids: Sequence[ImageGrid], spatial_merge_size: int) -> Float[Array, "patches channels"]:
        per_grid = []
        for grid in grids:
            taps = self.config.bilinear_taps(grid, spatial_merge_size)
            corners = self.table[taps.indices]
            # Blend in float32 even for low-precision tables. HF blends in the table dtype,
            # which only costs accuracy; the difference is far below bfloat16 resolution.
            blended = einsum(
                corners.astype(jnp.float32),
                jnp.asarray(taps.weights),
                "corners patches channels, corners patches -> patches channels",
            ).astype(self.table.dtype)
            per_grid.append(jnp.tile(blended, (grid.num_frames, 1)))
        return jnp.concatenate(per_grid, axis=0)
