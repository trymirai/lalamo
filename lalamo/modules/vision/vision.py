from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from lalamo.module import LalamoConfig

from .image_preprocessing import ImagePreprocessingConfig, PreprocessedImages
from .vision_encoder import VisionEncoderConfig

__all__ = ["VisionConfig"]


@dataclass(frozen=True)
class VisionConfig(LalamoConfig):
    encoder_config: VisionEncoderConfig
    preprocessing_config: ImagePreprocessingConfig

    def preprocess(self, images: Sequence[Image.Image]) -> PreprocessedImages:
        return self.preprocessing_config.preprocess(images, self.encoder_config)
