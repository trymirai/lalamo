from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

from lalamo.modules.vision import ImageGrid, ImagePreprocessingConfig

from .common import lalamo_encoder

ENCODER_CONFIG = replace(
    lalamo_encoder().config,
    patch_size=16,
)
CONFIG = ImagePreprocessingConfig(
    min_pixels=65_536,
    max_pixels=1_048_576,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
)


def _hf_processor() -> Qwen2VLImageProcessorPil:
    return Qwen2VLImageProcessorPil(
        size={"shortest_edge": CONFIG.min_pixels, "longest_edge": CONFIG.max_pixels},
        patch_size=ENCODER_CONFIG.patch_size,
        temporal_patch_size=ENCODER_CONFIG.temporal_patch_size,
        merge_size=ENCODER_CONFIG.spatial_merge_size,
        image_mean=list(CONFIG.image_mean),
        image_std=list(CONFIG.image_std),
    )


def _random_image(width: int, height: int, mode: str, seed: int) -> Image.Image:
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(pixels.squeeze(-1) if channels == 1 else pixels, mode=mode)


IMAGE_CASES = [
    pytest.param((300, 200, "RGB"), id="downscale-free-rgb"),
    pytest.param((50, 40, "RGB"), id="upscaled-to-min-pixels"),
    pytest.param((2000, 1500, "RGB"), id="downscaled-to-max-pixels"),
    pytest.param((640, 97, "RGBA"), id="wide-rgba"),
    pytest.param((128, 512, "L"), id="tall-grayscale"),
]


@pytest.mark.parametrize("case", IMAGE_CASES)
def test_matches_huggingface(case: tuple[int, int, str]) -> None:
    width, height, mode = case
    images = [_random_image(width, height, mode, seed=0), _random_image(height, width, mode, seed=1)]

    result = CONFIG.preprocess(images, ENCODER_CONFIG)
    reference = _hf_processor()(images=images, return_tensors="np")

    assert result.grids == tuple(ImageGrid(*row) for row in reference["image_grid_thw"].tolist())
    np.testing.assert_allclose(result.patches, reference["pixel_values"], atol=1e-6, rtol=0)


def test_rejects_extreme_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="Aspect ratio"):
        CONFIG.preprocess([Image.new("RGB", (2010, 10))], ENCODER_CONFIG)
