from pathlib import Path

import jax.numpy as jnp
import pytest

from lalamo.exportable import Exportable
from lalamo.model_import.loaders.vision import load_hf_vision
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.modules.vision import Vision
from lalamo.utils.sharding import ShardingConfig


def _assert_same_arrays(result: Exportable, reference: Exportable) -> None:
    result_arrays = Exportable.export(result).arrays
    reference_arrays = Exportable.export(reference).arrays
    assert result_arrays.keys() == reference_arrays.keys()
    mismatched = [name for name in result_arrays if not jnp.array_equal(result_arrays[name], reference_arrays[name])]
    assert not mismatched, f"Arrays differ: {mismatched[:5]}"


@pytest.mark.parametrize("repo", ["Qwen/Qwen3.5-0.8B"])
def test_standalone_vision_export_loads(repo: str, tmp_path: Path) -> None:
    sharding_config = ShardingConfig.replicated()
    vision = load_hf_vision(HuggingFaceOrigin(repo), sharding_config=sharding_config, dtype=jnp.bfloat16)
    vision.save(tmp_path / "vision")
    restored_vision = Vision.load(tmp_path / "vision", sharding_config)
    assert restored_vision.config == vision.config
    _assert_same_arrays(restored_vision, vision)
