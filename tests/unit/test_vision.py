import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cattrs import ClassValidationError
from PIL import Image

from lalamo.initializer import RandomInitializer
from lalamo.model_import.loaders.vision import load_hf_vision
from lalamo.model_import.model_configs.huggingface.qwen3_5_vision import HFQwen35VisionConfig
from lalamo.model_import.origins import LocalOrigin
from lalamo.module import Keychain
from lalamo.modules.vision import ImagePreprocessingConfig, MRopeConfig, Vision, VisionConfig
from lalamo.safetensors import safe_write
from lalamo.utils.sharding import ShardingConfig
from tests.unit.modules.vision.common import TINY_ENCODER_CONFIG, TINY_VISION_CONFIG, tiny_hf_vision_model


def _vision_config() -> VisionConfig:
    return VisionConfig(
        encoder_config=TINY_ENCODER_CONFIG,
        preprocessing_config=ImagePreprocessingConfig(
            min_pixels=256,
            max_pixels=4096,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ),
        mrope_config=MRopeConfig(sections=(2, 1, 1), interleaved=True),
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


@pytest.mark.parametrize("interleaved", [True, False])
def test_vision_export_preserves_hf_mrope(tmp_path: Path, interleaved: bool) -> None:
    sections = [12, 10, 10]
    rope_parameters = {"mrope_section": sections, "mrope_interleaved": interleaved}
    model = tiny_hf_vision_model()
    config = {
        "text_config": {"rope_parameters": rope_parameters},
        "vision_config": {**model.config.to_dict(), "model_type": "qwen3_5", "deepstack_visual_indexes": []},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 256, "longest_edge": 4096},
                "patch_size": model.config.patch_size,
                "temporal_patch_size": model.config.temporal_patch_size,
                "merge_size": model.config.spatial_merge_size,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            }
        )
    )
    with (tmp_path / "model.safetensors").open("wb") as weights_file:
        weights = {
            f"model.visual.{name}": jnp.asarray(tensor.detach().numpy()) for name, tensor in model.state_dict().items()
        }
        safe_write(weights_file, weights)

    origin = LocalOrigin(str(tmp_path), weight_files=("model.safetensors",))
    vision = load_hf_vision(
        origin, sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:1]), dtype=jnp.float32
    )
    vision.save(tmp_path / "vision")

    exported_config = json.loads((tmp_path / "vision" / "config.json").read_text())
    assert exported_config["mrope_config"] == {"sections": sections, "interleaved": interleaved}
    restored_config = VisionConfig.from_json(exported_config)
    assert restored_config.mrope_config == MRopeConfig(sections=(12, 10, 10), interleaved=interleaved)


@pytest.mark.parametrize("missing_field", ["mrope_section", "mrope_interleaved"])
def test_hf_vision_config_requires_mrope_fields(tmp_path: Path, missing_field: str) -> None:
    rope_parameters = {"mrope_section": [12, 10, 10], "mrope_interleaved": True}
    del rope_parameters[missing_field]
    config = {
        "text_config": {"rope_parameters": rope_parameters},
        "vision_config": {**TINY_VISION_CONFIG.to_dict(), "model_type": "qwen3_5", "deepstack_visual_indexes": []},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=missing_field):
        HFQwen35VisionConfig.from_json(config_path)


def test_vision_config_loads_flat_mrope_fields() -> None:
    config = json.loads(json.dumps(_vision_config().to_json()))
    del config["mrope_config"]
    config.update(mrope_sections=[12, 10, 10], mrope_interleaved=False)
    assert VisionConfig.from_json(config).mrope_config == MRopeConfig(sections=(12, 10, 10), interleaved=False)


def test_vision_config_requires_mrope_fields() -> None:
    config = json.loads(json.dumps(_vision_config().to_json()))
    del config["mrope_config"]
    with pytest.raises(ValueError, match="Missing MRoPE settings"):
        VisionConfig.from_json(config)


@pytest.mark.parametrize("field_name", ["is_causal", "has_gate"])
def test_vision_rejects_unsupported_attention_in_json(field_name: str) -> None:
    config = json.loads(json.dumps(_vision_config().to_json()))
    config["encoder_config"]["layer_configs"][0]["attention_config"][field_name] = True
    with pytest.raises(ClassValidationError):
        VisionConfig.from_json(config)
