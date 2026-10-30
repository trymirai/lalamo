import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from einops import rearrange
from jaxtyping import Float
from PIL import Image

from lalamo.module import LalamoConfig

from .image_grid import ImageGrid

__all__ = [
    "ImagePreprocessingConfig",
    "PreprocessedImages",
]


class PreprocessedImages(NamedTuple):
    patches: Float[np.ndarray, "patches patch_channels"]
    grids: tuple[ImageGrid, ...]


@dataclass(frozen=True)
class ImagePreprocessingConfig(LalamoConfig):
    """Host-side image preprocessing, a NumPy/Pillow port of HF's ``Qwen2VLImageProcessor`` (PIL backend).

    The steps are: convert to RGB, resize with bicubic filtering to the nearest size whose
    sides are multiples of ``patch_size * spatial_merge_size`` (keeping the pixel count within
    ``[min_pixels, max_pixels]``), scale to ``[0, 1]``, normalise per channel, repeat the still
    image ``temporal_patch_size`` times along time, and cut the result into flattened patches.

    This runs on the CPU before anything reaches the accelerator. It produces exactly what
    ``VisionEncoder`` consumes, and it is also a precise spec of what an on-device runtime has
    to reproduce.
    """

    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    min_pixels: int
    max_pixels: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std need one value per RGB channel.")

    @property
    def resize_factor(self) -> int:
        return self.patch_size * self.spatial_merge_size

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        """HF's ``smart_resize``: nearest factor-aligned size, then rescaled into the pixel budget."""
        if max(height, width) / min(height, width) > 200:
            raise ValueError(f"Aspect ratio of a {height}x{width} image exceeds 200.")
        factor = self.resize_factor
        target_height = round(height / factor) * factor
        target_width = round(width / factor) * factor
        if target_height * target_width > self.max_pixels:
            shrink = math.sqrt(height * width / self.max_pixels)
            target_height = max(factor, math.floor(height / shrink / factor) * factor)
            target_width = max(factor, math.floor(width / shrink / factor) * factor)
        elif target_height * target_width < self.min_pixels:
            grow = math.sqrt(self.min_pixels / (height * width))
            target_height = math.ceil(height * grow / factor) * factor
            target_width = math.ceil(width * grow / factor) * factor
        return target_height, target_width

    def grid_for(self, height: int, width: int) -> ImageGrid:
        """Patch grid an image of this size produces, without touching any pixels."""
        target_height, target_width = self.target_size(height, width)
        return ImageGrid(
            num_frames=1,
            height_in_patches=target_height // self.patch_size,
            width_in_patches=target_width // self.patch_size,
        )

    def _normalized_pixels(self, image: Image.Image) -> Float[np.ndarray, "height width channels"]:
        rgb_image = image.convert("RGB")
        target_height, target_width = self.target_size(rgb_image.height, rgb_image.width)
        resized = rgb_image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)
        # Same arithmetic as HF: rescale in float64, downcast, then normalise in float32.
        scaled = (np.asarray(resized, dtype=np.float64) * (1 / 255)).astype(np.float32)
        mean = np.asarray(self.image_mean, dtype=np.float32)
        std = np.asarray(self.image_std, dtype=np.float32)
        return (scaled - mean) / std

    def _patchify(self, pixels: Float[np.ndarray, "height width channels"]) -> PreprocessedImages:
        height, width, _ = pixels.shape
        # A still image is a one-step clip; the temporal patch sees the same frame repeated.
        frames = np.repeat(pixels[None], self.temporal_patch_size, axis=0)
        patches = rearrange(
            frames,
            "(grid_t t) (block_h merge_h p_h) (block_w merge_w p_w) c"
            " -> (grid_t block_h block_w merge_h merge_w) (c t p_h p_w)",
            t=self.temporal_patch_size,
            merge_h=self.spatial_merge_size,
            merge_w=self.spatial_merge_size,
            p_h=self.patch_size,
            p_w=self.patch_size,
        )
        grid = ImageGrid(
            num_frames=1,
            height_in_patches=height // self.patch_size,
            width_in_patches=width // self.patch_size,
        )
        return PreprocessedImages(patches=np.ascontiguousarray(patches), grids=(grid,))

    def preprocess(self, images: Sequence[Image.Image]) -> PreprocessedImages:
        if not images:
            raise ValueError("Expected at least one image.")
        per_image = [self._patchify(self._normalized_pixels(image)) for image in images]
        return PreprocessedImages(
            patches=np.concatenate([result.patches for result in per_image], axis=0),
            grids=tuple(grid for result in per_image for grid in result.grids),
        )
