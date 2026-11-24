import json

import pytest
from cattrs import ClassValidationError

from lalamo.modules.vision import ImagePreprocessingConfig, VisionConfig
from tests.unit.modules.vision.common import TINY_ENCODER_CONFIG


def _vision_config() -> VisionConfig:
    return VisionConfig(
        encoder_config=TINY_ENCODER_CONFIG,
        preprocessing_config=ImagePreprocessingConfig(
            min_pixels=256,
            max_pixels=4096,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ),
    )


@pytest.mark.parametrize("field_name", ["is_causal", "has_gate"])
def test_vision_rejects_unsupported_attention_in_json(field_name: str) -> None:
    config = json.loads(json.dumps(_vision_config().to_json()))
    config["encoder_config"]["layer_configs"][0]["attention_config"][field_name] = True
    with pytest.raises(ClassValidationError):
        VisionConfig.from_json(config)
