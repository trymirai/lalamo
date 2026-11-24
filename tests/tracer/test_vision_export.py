from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lalamo.model_import.common import import_model
from lalamo.models import LanguageModel
from lalamo.utils.sharding import ShardingConfig


def test_multimodal_model_export_loads(tmp_path: Path) -> None:
    sharding_config = ShardingConfig.replicated()
    model = import_model("Qwen/Qwen3.5-0.8B", sharding_config=sharding_config, dtype=jnp.bfloat16).model
    assert isinstance(model, LanguageModel)
    assert model.vision_encoder is not None
    model.save(tmp_path)
    restored_model = LanguageModel.load(tmp_path, sharding_config)
    assert restored_model.config == model.config
    expected = model.export().arrays
    actual = restored_model.export().arrays
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
