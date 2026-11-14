import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from lalamo.initializer import Initializer
from lalamo.model import BaseModel
from lalamo.module import LalamoConfig

from .image_preprocessing import ImagePreprocessingConfig, PreprocessedImages
from .vision_encoder import VisionEncoder, VisionEncoderConfig

__all__ = ["Vision", "VisionConfig"]


@dataclass(frozen=True)
class VisionConfig(LalamoConfig):
    encoder_config: VisionEncoderConfig
    preprocessing_config: ImagePreprocessingConfig

    def preprocess(self, images: Sequence[Image.Image]) -> PreprocessedImages:
        return self.preprocessing_config.preprocess(images, self.encoder_config)

    def init(self, initializer: Initializer) -> "Vision":
        return Vision(
            config=self,
            sharding_config=initializer.sharding_config,
            encoder=self.encoder_config.init(initializer),
        )


class Vision(BaseModel[VisionConfig]):
    encoder: VisionEncoder

    @staticmethod
    def init_from_directory(directory: Path | str, initializer: Initializer) -> "Vision":
        with (Path(directory) / "config.json").open() as config_file:
            config = VisionConfig.from_json(json.load(config_file))
        return config.init(initializer)
