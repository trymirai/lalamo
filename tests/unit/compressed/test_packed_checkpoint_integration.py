import os
from pathlib import Path

import jax
import pytest

from lalamo.models.language_model import LanguageModel
from lalamo.safetensors import safe_read
from lalamo.utils.sharding import ShardingConfig
from tests.unit.compressed.test_packed_checkpoint import assert_loaded_every_saved_tensor

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif("PACKED_CHECKPOINT_DIRECTORY" not in os.environ, reason="requires a packed checkpoint package"),
]


def test_packed_checkpoint_loads_every_saved_tensor() -> None:
    directory = Path(os.environ["PACKED_CHECKPOINT_DIRECTORY"])
    # The test session exposes eight CPU devices; replicating the checkpoint on each would hold eight copies.
    config = ShardingConfig.replicated(jax.devices()[:1])
    with jax.set_mesh(config.mesh), (directory / "model.safetensors").open("rb") as stream:
        model = LanguageModel.load(directory, config)
        _, saved = safe_read(stream)
        assert_loaded_every_saved_tensor(model, saved)
