import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cattrs import ClassValidationError
from PIL import Image

from lalamo.initializer import RandomInitializer
from lalamo.module import Keychain
from lalamo.modules.vision import ImagePreprocessingConfig, Vision, VisionConfig
from lalamo.utils.sharding import ShardingConfig
from tests.unit.modules.vision.common import lalamo_encoder


def _vision_config() -> VisionConfig:
    return VisionConfig(
        encoder_config=lalamo_encoder().config,
        preprocessing_config=ImagePreprocessingConfig(
            min_pixels=256,
            max_pixels=4096,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ),
    )


def test_vision_directory_round_trip_preserves_embeddings(tmp_path: Path) -> None:
    sharding = ShardingConfig.replicated(jax.devices("cpu")[:1])
    initializer = RandomInitializer(jnp.float32, sharding, key=jax.random.key(0))
    vision = _vision_config().init(initializer)
    vision.save(tmp_path / "vision")
    config = json.loads((tmp_path / "vision" / "config.json").read_text())
    encoder_config = config["encoder_config"]
    assert encoder_config["rope_config"]["type"] == "VisionRoPEConfig"
    assert all(layer["attention_config"]["type"] == "AttentionConfig" for layer in encoder_config["layer_configs"])
    restored = Vision.load(tmp_path / "vision", sharding)
    assert restored.config == vision.config
    assert {path.name for path in (tmp_path / "vision").iterdir()} == {"config.json", "model.safetensors"}

    image = Image.fromarray(np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3))
    preprocessed = restored.config.preprocess([image])
    patches = jnp.asarray(preprocessed.patches)
    keychain = Keychain.init(0, sharding_config=sharding)
    reference = vision.encoder(patches, preprocessed.grids, keychain=keychain).outputs
    result = restored.encoder(patches, preprocessed.grids, keychain=keychain).outputs
    assert jnp.array_equal(result, reference)


@pytest.mark.parametrize("field_name", ["is_causal", "has_gate"])
def test_vision_rejects_unsupported_attention_in_json(field_name: str) -> None:
    config = json.loads(json.dumps(_vision_config().to_json()))
    config["encoder_config"]["layer_configs"][0]["attention_config"][field_name] = True
    with pytest.raises(ClassValidationError):
        VisionConfig.from_json(config)
