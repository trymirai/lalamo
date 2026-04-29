import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.initializer import RandomInitializer
from lalamo.module import ShardingAxis
from lalamo.modules.rope import (
    LinearScalingRoPEConfig,
    LlamaRoPEConfig,
    PositionalEmbeddings,
    RoPE,
    RoPEConfig,
    UnscaledRoPEConfig,
    YARNRoPEConfig,
)
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from tests.common import assert_close

HEAD_DIM = 4
NUM_TIMESTEPS = 8


def _rope(config: RoPEConfig | None = None) -> RoPE:
    if config is None:
        config = UnscaledRoPEConfig(base=10_000.0, max_sequence_length=NUM_TIMESTEPS)
    return config.init(
        RandomInitializer(dtype=jnp.float32, key=jax.random.key(0)),
        head_dim=HEAD_DIM,
        num_timesteps=NUM_TIMESTEPS,
    )


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_heads(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA, None)))


def _apply_reference(embeddings: PositionalEmbeddings, heads: Array) -> Array:
    embeddings = PositionalEmbeddings(
        cosines=jnp.asarray(jax.device_get(embeddings.cosines)),
        sines=jnp.asarray(jax.device_get(embeddings.sines)),
    )
    heads = jnp.asarray(jax.device_get(heads))
    rotated = heads[..., : embeddings.head_dim]
    rotated_half = jnp.concatenate(
        (
            -rotated[..., embeddings.head_dim // 2 :],
            rotated[..., : embeddings.head_dim // 2],
        ),
        axis=-1,
    )
    result = rotated * embeddings.cosines.astype(heads.dtype) + rotated_half * embeddings.sines.astype(heads.dtype)
    if heads.shape[-1] == embeddings.head_dim:
        return result
    return jnp.concatenate((result, heads[..., embeddings.head_dim :]), axis=-1)


def _select_reference(table: Array, timesteps: Array) -> Array:
    return table[jnp.asarray(jax.device_get(timesteps))]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(UnscaledRoPEConfig(base=10_000.0, max_sequence_length=NUM_TIMESTEPS), id="unscaled"),
        pytest.param(
            LinearScalingRoPEConfig(base=10_000.0, max_sequence_length=NUM_TIMESTEPS, scaling_factor=2.0),
            id="linear-scaling",
        ),
        pytest.param(
            LlamaRoPEConfig(
                base=10_000.0,
                max_sequence_length=NUM_TIMESTEPS,
                scaling_factor=4.0,
                original_context_length=NUM_TIMESTEPS,
                low_frequency_factor=1.0,
                high_frequency_factor=4.0,
            ),
            id="llama",
        ),
        pytest.param(
            YARNRoPEConfig(
                base=10_000.0,
                max_sequence_length=NUM_TIMESTEPS,
                scaling_factor=2.0,
                original_context_length=NUM_TIMESTEPS,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=True,
            ),
            id="yarn",
        ),
    ],
)
def test_rope_config_init_produces_finite_tables(config: RoPEConfig) -> None:
    rope = _rope(config)

    assert rope.sines.shape == (NUM_TIMESTEPS, HEAD_DIM)
    assert rope.cosines.shape == (NUM_TIMESTEPS, HEAD_DIM)
    assert rope.sines.dtype == jnp.float32
    assert rope.cosines.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(rope.sines))
    assert jnp.all(jnp.isfinite(rope.cosines))


def test_rope_selects_timesteps_under_jit() -> None:
    rope = _rope()
    timesteps = jnp.array([0, 1, 3, 5], dtype=jnp.int32)

    embeddings = eqx.filter_jit(lambda rope, timesteps: rope(timesteps))(rope, timesteps)

    _assert_close(result=embeddings.sines, reference=_select_reference(rope.sines, timesteps))
    _assert_close(result=embeddings.cosines, reference=_select_reference(rope.cosines, timesteps))


def test_rope_call_vmapped_over_batches_preserves_batch_sharding(fake_mesh: Mesh) -> None:
    rope = _rope()
    timesteps = jnp.array(
        [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
        ],
        dtype=jnp.int32,
    )
    timesteps = jax.device_put(timesteps, make_sharding((ShardingAxis.DATA, None)))

    embeddings = call_vmapped(rope, timesteps, added_sharding_axis=ShardingAxis.DATA)

    _assert_close(result=embeddings.sines, reference=_select_reference(rope.sines, timesteps))
    _assert_close(result=embeddings.cosines, reference=_select_reference(rope.cosines, timesteps))
    _assert_named_sharding(embeddings.sines.sharding, fake_mesh)
    _assert_named_sharding(embeddings.cosines.sharding, fake_mesh)
    assert embeddings.sines.sharding == make_sharding((ShardingAxis.DATA, None, None))
    assert embeddings.cosines.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_positional_embeddings_apply_matches_reference_and_preserves_sharding(fake_mesh: Mesh) -> None:
    rope = _rope()
    timesteps = jnp.array([0, 1, 3, 5], dtype=jnp.int32)
    embeddings = rope(timesteps)
    heads = _sharded_heads(jnp.arange(4 * 6, dtype=jnp.float32).reshape(4, 6) / 10)

    result = embeddings.apply(heads)

    _assert_close(result=result, reference=_apply_reference(embeddings, heads))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))


def test_positional_embeddings_apply_output_dtype_matches_input_dtype(fake_mesh: Mesh) -> None:
    rope = _rope()
    timesteps = jnp.array([0, 1, 3, 5], dtype=jnp.int32)
    embeddings = rope(timesteps)
    heads = _sharded_heads(jnp.arange(4 * 6, dtype=jnp.bfloat16).reshape(4, 6) / 10)

    result = embeddings.apply(heads)

    assert result.dtype == heads.dtype
    _assert_close(result=result, reference=_apply_reference(embeddings, heads))
    _assert_named_sharding(result.sharding, fake_mesh)


def test_positional_embeddings_apply_rejects_too_small_head_dim() -> None:
    embeddings = _rope()(jnp.array([0, 1], dtype=jnp.int32))

    with pytest.raises(ValueError, match="exceeds input head_dim"):
        embeddings.apply(jnp.zeros((2, HEAD_DIM - 1), dtype=jnp.float32))


def test_rope_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _rope()
    table_sharding = make_sharding((None, ShardingAxis.TENSOR))
    template = RoPE(
        config=original.config,
        sines=dummy_array(original.sines.shape, original.sines.dtype, table_sharding),
        cosines=dummy_array(original.cosines.shape, original.cosines.dtype, table_sharding),
    )
    timesteps = jnp.array([0, 1, 3, 5], dtype=jnp.int32)

    restored = template.load_exported(original.export())
    embeddings = restored(timesteps)

    assert restored.sines.sharding == template.sines.sharding
    assert restored.cosines.sharding == template.cosines.sharding
    _assert_named_sharding(restored.sines.sharding, fake_mesh)
    _assert_named_sharding(restored.cosines.sharding, fake_mesh)
    _assert_close(result=embeddings.sines, reference=_select_reference(original.sines, timesteps))
    _assert_close(result=embeddings.cosines, reference=_select_reference(original.cosines, timesteps))
