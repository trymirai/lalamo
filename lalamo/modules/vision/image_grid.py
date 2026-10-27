from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from jaxtyping import Int

__all__ = [
    "ImageGrid",
    "PatchCoordinates",
]


class PatchCoordinates(NamedTuple):
    rows: Int[np.ndarray, " patches"]
    cols: Int[np.ndarray, " patches"]


@dataclass(frozen=True)
class ImageGrid:
    """Size of one image (or video clip) measured in vision patches.

    HuggingFace calls this ``grid_thw``. The fields are plain Python ints on purpose: they decide
    array shapes inside the encoder, so they have to be static under ``jax.jit``. Every distinct
    grid compiles its own program, which is fine because the preprocessor snaps image sizes to
    multiples of ``patch_size * spatial_merge_size`` and the set of shapes stays small in practice.
    """

    num_frames: int
    height_in_patches: int
    width_in_patches: int

    def __post_init__(self) -> None:
        if min(self.num_frames, self.height_in_patches, self.width_in_patches) < 1:
            raise ValueError(f"All grid dimensions must be positive, got {self}.")

    @property
    def patches_per_frame(self) -> int:
        return self.height_in_patches * self.width_in_patches

    @property
    def num_patches(self) -> int:
        return self.num_frames * self.patches_per_frame

    def require_mergeable(self, spatial_merge_size: int) -> None:
        if self.height_in_patches % spatial_merge_size or self.width_in_patches % spatial_merge_size:
            raise ValueError(
                f"Grid {self.height_in_patches}x{self.width_in_patches} is not divisible by the"
                f" spatial merge size {spatial_merge_size}.",
            )

    def num_merged_tokens(self, spatial_merge_size: int) -> int:
        self.require_mergeable(spatial_merge_size)
        return self.num_patches // spatial_merge_size**2

    def patch_coordinates(self, spatial_merge_size: int) -> PatchCoordinates:
        """Row and column of every patch in one frame, listed in the order the encoder receives them.

        Patches do not arrive in raster order. The preprocessor keeps every ``merge x merge`` block
        of neighbouring patches contiguous, so that the patch merger at the end of the encoder can
        fuse a block with a plain reshape. The visiting order is therefore: block row, block column,
        row inside the block, column inside the block.
        """
        self.require_mergeable(spatial_merge_size)
        block_rows, block_cols, rows_in_block, cols_in_block = np.meshgrid(
            np.arange(self.height_in_patches // spatial_merge_size),
            np.arange(self.width_in_patches // spatial_merge_size),
            np.arange(spatial_merge_size),
            np.arange(spatial_merge_size),
            indexing="ij",
        )
        return PatchCoordinates(
            rows=(block_rows * spatial_merge_size + rows_in_block).reshape(-1),
            cols=(block_cols * spatial_merge_size + cols_in_block).reshape(-1),
        )
