from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from lalamo.module import LalamoConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.utils.registry_abc import RegistryABC

from .image_grid import ImageGrid

__all__ = ["VisionRoPEConfig"]


@dataclass(frozen=True)
class VisionRoPEConfig(LalamoConfig, RegistryABC):
    """2D RoPE: the first half of each head rotates with the patch row, the second half with the column."""

    base: float
    head_dim: int

    def __post_init__(self) -> None:
        # Two axes, each made of (cos, sin) pairs.
        if self.head_dim % 4:
            raise ValueError(f"Vision RoPE head_dim must be divisible by 4, got {self.head_dim}.")

    def compute_positional_embeddings(
        self,
        grids: Sequence[ImageGrid],
        spatial_merge_size: int,
    ) -> PositionalEmbeddings:
        per_grid_coordinates = [(grid.patch_coordinates(spatial_merge_size), grid.num_frames) for grid in grids]
        # Frames of a clip share the spatial layout, so their coordinates simply repeat.
        rows = np.concatenate([np.tile(coordinates.rows, frames) for coordinates, frames in per_grid_coordinates])
        cols = np.concatenate([np.tile(coordinates.cols, frames) for coordinates, frames in per_grid_coordinates])

        axis_dim = self.head_dim // 2
        # Keep the frequencies in float32, matching HF's inv_freq buffer.
        inverse_frequencies = 1.0 / (self.base ** (jnp.arange(0, axis_dim, 2, dtype=jnp.float32) / axis_dim))
        frequencies = jnp.concatenate(
            [
                jnp.outer(jnp.asarray(rows, dtype=jnp.float32), inverse_frequencies),
                jnp.outer(jnp.asarray(cols, dtype=jnp.float32), inverse_frequencies),
            ],
            axis=-1,
        )
        angles = jnp.concatenate([frequencies, frequencies], axis=-1)
        return PositionalEmbeddings(cosines=jnp.cos(angles), sines=jnp.sin(angles))
