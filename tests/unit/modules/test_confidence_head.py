import json
from dataclasses import replace
from enum import Enum, auto
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.initializer import RandomInitializer
from lalamo.modules.linear import LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.speculator import Speculator
from lalamo.modules.speculators.confidence_head import ConfidenceHeadConfig, ConfidenceHeadTreeAttentionConfig
from lalamo.modules.speculators.dflash import DFlashDraftConfig, DFlashSpeculatorConfig
from lalamo.safetensors import safe_write
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import build_tiny_attention_decoder_config

CONFIDENCE_HEAD_CONFIG = ConfidenceHeadConfig(
    model_dim=4,
    pool_hidden_dim=2,
    edge_hidden_dim=8,
    logit_cap=50.0,
    max_depth=3,
    norm_config=NormalizationConfig(
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.FULL_LAYER,
        subtract_mean=True,
        has_biases=True,
    ),
    tree_attention_config=ConfidenceHeadTreeAttentionConfig(model_dim=8, num_layers=1, num_heads=2, hidden_dim=16),
)


class CheckpointFault(Enum):
    MISSING_TENSOR = auto()
    EXTRA_TENSOR = auto()
    SHAPE = auto()
    DTYPE = auto()
    MISSING_METADATA = auto()
    EXTRA_METADATA = auto()
    UNKNOWN_SPEC = auto()


@pytest.fixture
def speculator_config() -> DFlashSpeculatorConfig:
    transformer = build_tiny_attention_decoder_config((None,)).transformer_config
    (layer_config,) = transformer.layer_configs
    rope_config = layer_config.rope_config
    assert rope_config is not None
    return DFlashSpeculatorConfig(
        draft_config=DFlashDraftConfig(
            model_dim=transformer.model_dim,
            hidden_dim=transformer.hidden_dim,
            block_size=4,
            mask_token_id=31,
            target_layer_ids=(0,),
            num_target_layers=1,
            vocab_size=32,
            context_projection_config=LinearConfig(),
            context_norm_config=CONFIDENCE_HEAD_CONFIG.norm_config,
            rope_config=rope_config,
            layer_configs=transformer.layer_configs,
            output_norm_config=CONFIDENCE_HEAD_CONFIG.norm_config,
        ),
        weaver_config=None,
        confidence_head_config=CONFIDENCE_HEAD_CONFIG,
    )


@pytest.mark.parametrize(
    "confidence_head_config",
    [None, replace(CONFIDENCE_HEAD_CONFIG, tree_attention_config=None), CONFIDENCE_HEAD_CONFIG],
    ids=["none", "pool", "tree"],
)
def test_speculator_roundtrip_preserves_confidence_head(
    speculator_config: DFlashSpeculatorConfig,
    tmp_path: Path,
    confidence_head_config: ConfidenceHeadConfig | None,
) -> None:
    model = replace(speculator_config, confidence_head_config=confidence_head_config).init(
        RandomInitializer(jnp.bfloat16, ShardingConfig.replicated(), key=jax.random.key(0)),
    )
    model.save(tmp_path)
    restored = Speculator.load(tmp_path, model.sharding_config)
    expected, actual = model.export(), restored.export()
    assert restored.config == model.config
    assert actual.metadata == expected.metadata
    assert actual.arrays.keys() == expected.arrays.keys()
    for name, array in expected.arrays.items():
        np.testing.assert_array_equal(actual.arrays[name], array, strict=True)


@pytest.mark.parametrize("fault", CheckpointFault)
def test_speculator_rejects_malformed_confidence_head(
    speculator_config: DFlashSpeculatorConfig, tmp_path: Path, fault: CheckpointFault
) -> None:
    speculator = speculator_config.init(
        RandomInitializer(jnp.bfloat16, ShardingConfig.replicated(), key=jax.random.key(0)),
    )
    speculator.save(tmp_path)
    exported = speculator.export()
    arrays = dict(exported.arrays)
    metadata = dict(exported.metadata)
    tensor_name = "confidence_head.pool_set.entry.weights"
    spec_name = "confidence_head.draft_projection.weights.spec"
    match fault:
        case CheckpointFault.MISSING_TENSOR:
            del arrays[tensor_name]
        case CheckpointFault.EXTRA_TENSOR:
            arrays["confidence_head.unexpected"] = jnp.zeros((1,))
        case CheckpointFault.SHAPE:
            arrays[tensor_name] = arrays[tensor_name][:, :-1]
        case CheckpointFault.DTYPE:
            arrays[tensor_name] = arrays[tensor_name].astype(jnp.int32)
        case CheckpointFault.MISSING_METADATA:
            del metadata[spec_name]
        case CheckpointFault.EXTRA_METADATA:
            metadata["confidence_head.unexpected"] = {}
        case CheckpointFault.UNKNOWN_SPEC:
            metadata[spec_name] = {"type": "UnknownSpec"}
    with (tmp_path / "model.safetensors").open("wb") as weights_file:
        safe_write(weights_file, arrays, {name: json.dumps(value) for name, value in metadata.items()})
    with pytest.raises((ValueError, AssertionError, KeyError)):
        Speculator.load(tmp_path, speculator.sharding_config)
