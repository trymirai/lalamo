from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from lalamo.module import LalamoConfig
from lalamo.modules.rope import PositionalEmbeddings

from .image_grid import ImageGrid

__all__ = ["VisionRoPEConfig"]


@dataclass(frozen=True)
class VisionRoPEConfig(LalamoConfig):
    """Two-dimensional rotary embedding used inside the vision transformer.

    Every patch carries a (row, column) position. The head is split in two halves: the first
    half rotates with the row, the second with the column. Each half uses the usual
    ``base ** (-2i / d)`` frequency ladder with ``d = head_dim / 2``.

    The result is returned as the same ``PositionalEmbeddings`` the text decoder uses, laid out
    as ``[row_freqs, col_freqs, row_freqs, col_freqs]``. That layout makes the standard
    rotate-half application (``PositionalEmbeddings.apply``) produce exactly the rotation HF
    performs in ``apply_rotary_pos_emb_vision``.

    There are no learned parameters, so this is a config without a matching module.
    """

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
        # Computed in float32 on purpose, mirroring HF's float32 inv_freq buffer bit for bit.
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
