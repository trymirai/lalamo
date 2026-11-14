import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from einops import repeat
from jaxtyping import Float
from PIL import Image

from lalamo.module import LalamoConfig

from .image_grid import ImageGrid
from .vision_encoder import VisionEncoderConfig

__all__ = [
    "ImagePreprocessingConfig",
    "PreprocessedImages",
]


class PreprocessedImages(NamedTuple):
    patches: Float[np.ndarray, "patches patch_channels"]
    grids: tuple[ImageGrid, ...]


@dataclass(frozen=True)
class ImagePreprocessingConfig(LalamoConfig):
    """A NumPy/Pillow port of HF's ``Qwen2VLImageProcessorPil``: resize, normalise, and cut into patches."""

    min_pixels: int
    max_pixels: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    def preprocess(self, images: Sequence[Image.Image], encoder_config: VisionEncoderConfig) -> PreprocessedImages:
        if not images:
            raise ValueError("Expected at least one image.")
        merge_size = encoder_config.spatial_merge_size
        factor = encoder_config.patch_size * merge_size
        mean = np.asarray(self.image_mean, dtype=np.float32)
        std = np.asarray(self.image_std, dtype=np.float32)
        patches, grids = [], []
        for image in images:
            width, height = image.size
            if max(height, width) / min(height, width) > 200:
                raise ValueError(f"Aspect ratio of a {height}x{width} image exceeds 200.")
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
            resized = image.convert("RGB").resize((target_width, target_height), resample=Image.Resampling.BICUBIC)
            # HF rescales in float64, downcasts, and then normalizes in float32.
            pixels = (np.asarray(resized, dtype=np.float64) * (1 / 255)).astype(np.float32)
            patches.append(
                repeat(
                    (pixels - mean) / std,
                    "(block_h merge_h p_h) (block_w merge_w p_w) c"
                    " -> (block_h block_w merge_h merge_w) (c temporal p_h p_w)",
                    temporal=encoder_config.temporal_patch_size,
                    merge_h=merge_size,
                    merge_w=merge_size,
                    p_h=encoder_config.patch_size,
                    p_w=encoder_config.patch_size,
                ),
            )
            grids.append(
                ImageGrid(1, target_height // encoder_config.patch_size, target_width // encoder_config.patch_size),
            )
        return PreprocessedImages(np.concatenate(patches), tuple(grids))
