from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from einops import rearrange
from jaxtyping import Int

__all__ = [
    "ImageGrid",
]


class PatchCoordinates(NamedTuple):
    rows: Int[np.ndarray, " patches"]
    cols: Int[np.ndarray, " patches"]


@dataclass(frozen=True)
class ImageGrid:
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

    def patch_coordinates(self, spatial_merge_size: int) -> PatchCoordinates:
        """Row and column of each patch, in HF's order: each ``merge x merge`` block of patches is contiguous."""
        if self.height_in_patches % spatial_merge_size or self.width_in_patches % spatial_merge_size:
            raise ValueError(
                f"Grid {self.height_in_patches}x{self.width_in_patches} is not divisible by the"
                f" spatial merge size {spatial_merge_size}.",
            )

        rows, cols = rearrange(
            np.indices((self.height_in_patches, self.width_in_patches)),
            "coord (block_h merge_h) (block_w merge_w) -> coord (block_h block_w merge_h merge_w)",
            merge_h=spatial_merge_size,
            merge_w=spatial_merge_size,
        )
        return PatchCoordinates(rows, cols)
