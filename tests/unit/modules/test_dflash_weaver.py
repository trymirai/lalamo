from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lalamo.initializer import RandomInitializer
from lalamo.models import SpeculatorModel, SpeculatorModelConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules import (
    DenseMLPConfig,
    DFlashAttentionConfig,
    DFlashDraftConfig,
    DFlashDraftLayerConfig,
    DFlashDraftModel,
    LinearConfig,
    NormalizationConfig,
    UnscaledRoPEConfig,
    UpcastMode,
    Weaver,
    WeaverConfig,
)
from lalamo.modules.activations import SiLU

MODEL_DIM = 8
VOCAB_SIZE = 32
HEAD_DIM = 4
BLOCK_SIZE = 4
D_RANK = 16
NUM_HEADS = 2
DEPTH = 2

NORM_CONFIG = NormalizationConfig(
    epsilon=1e-5,
    scale_offset=None,
    upcast_mode=UpcastMode.ONLY_NORMALIZATION,
    subtract_mean=False,
)
WEAVER_NORM_CONFIG = NormalizationConfig(
    epsilon=1e-6,
    scale_offset=None,
    upcast_mode=UpcastMode.FULL_LAYER,
    subtract_mean=False,
    has_biases=True,
)


def initializer(seed: int) -> RandomInitializer:
    return RandomInitializer(
        default_dtype=jnp.float32,
        sharding_config=ShardingConfig.replicated(),
        key=jax.random.key(seed),
    )


def weaver_config() -> WeaverConfig:
    return WeaverConfig(
        d_model=MODEL_DIM,
        d_embed=MODEL_DIM,
        d_rank=D_RANK,
        num_layers=1,
        num_heads=NUM_HEADS,
        mlp_dim=32,
        k=BLOCK_SIZE - 1,
        candidate_pool_size=16,
        linear_config=LinearConfig(),
        norm_config=WEAVER_NORM_CONFIG,
    )


def draft_config() -> DFlashDraftConfig:
    mlp_config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    return DFlashDraftConfig(
        model_dim=MODEL_DIM,
        hidden_dim=16,
        block_size=BLOCK_SIZE,
        mask_token_id=VOCAB_SIZE - 1,
        target_layer_ids=(0, 1),
        num_target_layers=2,
        vocab_size=VOCAB_SIZE,
        context_projection_config=LinearConfig(),
        context_norm_config=NORM_CONFIG,
        layer_configs=(
            DFlashDraftLayerConfig(
                attention_config=DFlashAttentionConfig(
                    linear_config=LinearConfig(),
                    query_norm_config=NORM_CONFIG,
                    key_norm_config=NORM_CONFIG,
                    rope_config=UnscaledRoPEConfig(base=10_000.0, max_sequence_length=64, head_dim=HEAD_DIM),
                    num_heads=2,
                    num_key_value_heads=2,
                    head_dim=HEAD_DIM,
                    has_attention_biases=False,
                    has_output_biases=False,
                    sliding_window_size=None,
                    scale=HEAD_DIM**-0.5,
                ),
                input_norm_config=NORM_CONFIG,
                post_attention_norm_config=NORM_CONFIG,
                mlp_config=mlp_config,
            ),
        ),
        output_norm_config=NORM_CONFIG,
    )


def assert_export_round_trip(model: Weaver | DFlashDraftModel, reloaded_template: Weaver | DFlashDraftModel) -> None:
    exported = model.export()
    restored = reloaded_template.load_exported(exported)
    original_leaves = [leaf for leaf in jax.tree.leaves(model) if isinstance(leaf, jax.Array)]
    restored_leaves = [leaf for leaf in jax.tree.leaves(restored) if isinstance(leaf, jax.Array)]
    assert len(original_leaves) == len(restored_leaves)
    for original, restored_leaf in zip(original_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(original), np.asarray(restored_leaf))


def test_weaver_export_round_trip() -> None:
    config = weaver_config()
    model = config.init(initializer(0))
    template = config.init(initializer(1))
    assert_export_round_trip(model, template)


def test_weaver_prefix_and_step_forward() -> None:
    config = weaver_config()
    weaver = weaver_config().init(initializer(0))
    keychain = Keychain.init(0, sharding_config=ShardingConfig.replicated())
    batch = 1
    output_norm_features = jax.random.normal(jax.random.key(1), (batch, config.d_model))
    proposal_features = jax.random.normal(jax.random.key(2), (batch, DEPTH, config.d_model))

    prefix = weaver.prompt_prefix(output_norm_features, proposal_features, keychain=keychain)
    assert prefix.keys.shape == (config.num_layers, batch, DEPTH + 1, config.num_heads, config.head_dim)

    pool = config.candidate_pool_size
    lm_head = jax.random.normal(jax.random.key(3), (VOCAB_SIZE, config.d_model))
    embed_w = jax.random.normal(jax.random.key(4), (VOCAB_SIZE, config.d_embed))
    candidate_ids = jnp.arange(pool, dtype=jnp.int32)[None, :]
    candidate_scores = jax.random.normal(jax.random.key(5), (batch, pool))
    ancestor_keys = jnp.zeros((config.num_layers, batch, DEPTH, config.num_heads, config.head_dim))
    ancestor_values = jnp.zeros_like(ancestor_keys)
    ancestor_mask = jnp.zeros((batch, DEPTH), dtype=jnp.bool)

    logits, keys, _ = weaver.step(
        lm_head,
        embed_w,
        jnp.zeros((batch,), dtype=jnp.int32),
        candidate_ids,
        candidate_scores,
        prefix.keys,
        prefix.values,
        jnp.zeros((batch,), dtype=jnp.int32),
        ancestor_keys,
        ancestor_values,
        ancestor_mask,
        keychain=keychain,
    )
    assert logits.shape == (batch, pool)
    assert keys.shape == (config.num_layers, batch, config.num_heads, config.head_dim)
    assert bool(jnp.all(jnp.isfinite(logits)))


def test_dflash_draft_model_export_round_trip() -> None:
    config = draft_config()
    model = config.init(initializer(0))
    template = config.init(initializer(1))
    assert_export_round_trip(model, template)


def assert_model_disk_round_trip(model: SpeculatorModel, directory: Path) -> None:
    model.save(directory)
    assert (directory / "model.safetensors").exists()
    assert (directory / "config.json").exists()
    assert not (directory / "tokenizer.json").exists()

    restored = SpeculatorModel.load(directory, ShardingConfig.replicated(), jnp.float32)
    original_leaves = [leaf for leaf in jax.tree.leaves(model) if isinstance(leaf, jax.Array)]
    restored_leaves = [leaf for leaf in jax.tree.leaves(restored) if isinstance(leaf, jax.Array)]
    for original, restored_leaf in zip(original_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(original), np.asarray(restored_leaf))


def test_speculator_model_with_weaver_save_load(tmp_path: Path) -> None:
    config = SpeculatorModelConfig(draft_config=draft_config(), weaver_config=weaver_config())
    model = config.init(initializer(0))
    assert model.weaver is not None
    assert_model_disk_round_trip(model, tmp_path / "speculator")


def test_speculator_model_without_weaver_save_load(tmp_path: Path) -> None:
    config = SpeculatorModelConfig(draft_config=draft_config(), weaver_config=None)
    model = config.init(initializer(0))
    assert model.weaver is None
    assert_model_disk_round_trip(model, tmp_path / "speculator")
