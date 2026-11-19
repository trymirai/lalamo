import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from PIL import Image

from lalamo.initializer import Initializer
from lalamo.model import BaseModel
from lalamo.module import LalamoConfig
from lalamo.utils.json import JSON

from .image_preprocessing import ImagePreprocessingConfig, PreprocessedImages
from .vision_encoder import VisionEncoder, VisionEncoderConfig

__all__ = ["MRopeConfig", "Vision", "VisionConfig"]


@dataclass(frozen=True)
class MRopeConfig(LalamoConfig):
    sections: tuple[int, int, int]
    interleaved: bool


@dataclass(frozen=True)
class VisionConfig(LalamoConfig):
    encoder_config: VisionEncoderConfig
    preprocessing_config: ImagePreprocessingConfig
    mrope_config: MRopeConfig

    @classmethod
    def from_json(cls, json_object: JSON) -> Self:
        if isinstance(json_object, dict) and "mrope_config" not in json_object:
            if "mrope_sections" not in json_object or "mrope_interleaved" not in json_object:
                raise ValueError(
                    "Missing MRoPE settings in vision config. Re-export from the Hugging Face checkpoint."
                )
            config = dict(json_object)
            config["mrope_config"] = {
                "sections": config.pop("mrope_sections"),
                "interleaved": config.pop("mrope_interleaved"),
            }
            json_object = config
        return super().from_json(json_object)

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
