import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import numpy as np
from einops import rearrange
from jax import numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule

from .image_grid import ImageGrid

__all__ = [
    "InterpolatedPositionEmbedding",
    "InterpolatedPositionEmbeddingConfig",
]


@dataclass(frozen=True)
class InterpolatedPositionEmbeddingConfig(LalamoConfig):
    """Learned absolute position table, bilinearly resampled to the image's patch grid."""

    num_positions: int

    def __post_init__(self) -> None:
        if math.isqrt(self.num_positions) ** 2 != self.num_positions:
            raise ValueError(f"num_positions must be a perfect square, got {self.num_positions}.")

    def init(self, initializer: Initializer, model_dim: int) -> "InterpolatedPositionEmbedding":
        return InterpolatedPositionEmbedding(
            config=self,
            sharding_config=initializer.sharding_config,
            table=initializer.normal(0.02, (self.num_positions, model_dim)),
        )


class InterpolatedPositionEmbedding(LalamoModule[InterpolatedPositionEmbeddingConfig]):
    table: Float[Array, "positions channels"]

    @eqx.filter_jit
    def __call__(self, grids: Sequence[ImageGrid], spatial_merge_size: int) -> Float[Array, "patches channels"]:
        side = math.isqrt(self.config.num_positions)
        # HF blends in float32 before rounding back to the table dtype.
        table = rearrange(self.table.astype(jnp.float32), "(row col) channels -> row col channels", row=side, col=side)
        per_grid = []
        for grid in grids:
            rows, cols = grid.patch_coordinates(spatial_merge_size)
            coordinates = (
                np.linspace(0, side - 1, grid.height_in_patches, dtype=np.float32)[rows],
                np.linspace(0, side - 1, grid.width_in_patches, dtype=np.float32)[cols],
            )
            blended = jax.vmap(
                partial(map_coordinates, coordinates=coordinates, order=1, mode="nearest"),
                in_axes=2,
                out_axes=1,
            )(table).astype(self.table.dtype)
            per_grid.append(jnp.tile(blended, (grid.num_frames, 1)))
        return jnp.concatenate(per_grid, axis=0)
