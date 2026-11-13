from pathlib import Path

import jax.numpy as jnp
import pytest

from lalamo.exportable import Exportable
from lalamo.model import Model
from lalamo.model_import.common import import_model, import_vision_encoder
from lalamo.model_registry import ModelRegistry
from lalamo.models import LanguageModel, VisionLanguageModel, VisionLanguageModelConfig
from lalamo.modules import MRoPEConfig
from lalamo.utils.sharding import ShardingConfig


def _assert_same_arrays(result: Exportable, reference: Exportable, prefix: str = "") -> None:
    result_arrays = Exportable.export(result).arrays
    reference_arrays = Exportable.export(reference).arrays
    assert result_arrays.keys() == reference_arrays.keys()
    mismatched = [name for name in result_arrays if not jnp.array_equal(result_arrays[name], reference_arrays[name])]
    assert not mismatched, f"Arrays differ: {[prefix + name for name in mismatched][:5]}"


@pytest.mark.parametrize("repo", ["Qwen/Qwen3.5-0.8B"])
def test_vision_export_matches_text_and_encoder_imports(repo: str, tmp_path: Path) -> None:
    sharding_config = ShardingConfig.replicated()
    model = import_model(repo, sharding_config=sharding_config, dtype=jnp.bfloat16, include_vision=True).model
    assert isinstance(model, VisionLanguageModel)
    assert isinstance(model.config, VisionLanguageModelConfig)
    assert all(
        isinstance(layer_config.rope_config, MRoPEConfig)
        for layer_config in model.config.decoder_config.transformer_config.layer_configs
        if layer_config.rope_config is not None
    )

    text_model = import_model(repo, sharding_config=sharding_config, dtype=jnp.bfloat16).model
    assert isinstance(text_model, LanguageModel)
    _assert_same_arrays(model.decoder, text_model.decoder, "decoder.")
    del text_model

    spec = ModelRegistry.init().repo_to_model[repo]
    encoder = import_vision_encoder(spec, sharding_config=sharding_config, dtype=jnp.bfloat16)
    _assert_same_arrays(model.vision_encoder, encoder.encoder, "vision_encoder.")
    assert model.config.image_preprocessing_config == encoder.preprocessing_config
    del encoder

    model.save(tmp_path)
    restored = Model.load(tmp_path, sharding_config)
    assert isinstance(restored, VisionLanguageModel)
    assert restored.config == model.config
    _assert_same_arrays(restored, model)
