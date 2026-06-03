import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.initializer import EmptyInitializer, RandomInitializer
from lalamo.module import LogicalAxis
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
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config

HEAD_DIM = 4
NUM_TIMESTEPS = 8


def _rope(config: RoPEConfig | None = None) -> RoPE:
    if config is None:
        config = UnscaledRoPEConfig(base=10_000.0, max_sequence_length=NUM_TIMESTEPS, head_dim=HEAD_DIM)
    return config.init(
        RandomInitializer(
            default_dtype=jnp.float32, sharding_config=make_test_sharding_config(), key=jax.random.key(0)
        ),
    )


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_heads(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.BATCH, None)))


def _apply_reference(embeddings: PositionalEmbeddings, heads: Array) -> Array:
    embeddings = PositionalEmbeddings(
        cosines=jnp.asarray(jax.device_get(embeddings.cosines)),
        sines=jnp.asarray(jax.device_get(embeddings.sines)),
    )
    dtype = embeddings.cosines.dtype
    heads = jnp.asarray(jax.device_get(heads)).astype(dtype)
    rotated = heads[..., : embeddings.head_dim]
    rotated_half = jnp.concatenate(
        (
            -rotated[..., embeddings.head_dim // 2 :],
            rotated[..., : embeddings.head_dim // 2],
        ),
        axis=-1,
    )
    result = rotated * embeddings.cosines + rotated_half * embeddings.sines
    if heads.shape[-1] == embeddings.head_dim:
        return result
    return jnp.concatenate((result, heads[..., embeddings.head_dim :]), axis=-1)


def _select_reference(table: Array, timesteps: Array) -> Array:
    return table[jnp.asarray(jax.device_get(timesteps))]


def _full_table(rope: RoPE) -> PositionalEmbeddings:
    return rope.config.compute_positional_embeddings(jnp.arange(NUM_TIMESTEPS, dtype=jnp.int32))


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            UnscaledRoPEConfig(base=10_000.0, max_sequence_length=NUM_TIMESTEPS, head_dim=HEAD_DIM),
            id="unscaled",
        ),
        pytest.param(
            LinearScalingRoPEConfig(
                base=10_000.0,
                max_sequence_length=NUM_TIMESTEPS,
                head_dim=HEAD_DIM,
                scaling_factor=2.0,
            ),
            id="linear-scaling",
        ),
        pytest.param(
            LlamaRoPEConfig(
                base=10_000.0,
                max_sequence_length=NUM_TIMESTEPS,
                head_dim=HEAD_DIM,
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
                head_dim=HEAD_DIM,
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
    embeddings = _full_table(_rope(config))

    assert embeddings.sines.shape == (NUM_TIMESTEPS, HEAD_DIM)
    assert embeddings.cosines.shape == (NUM_TIMESTEPS, HEAD_DIM)
    assert embeddings.sines.dtype == jnp.float32
    assert embeddings.cosines.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(embeddings.sines))
    assert jnp.all(jnp.isfinite(embeddings.cosines))


def test_rope_selects_timesteps_under_jit() -> None:
    rope = _rope()
    timesteps = jnp.array([0, 1, 3, 5], dtype=jnp.int32)

    embeddings = eqx.filter_jit(lambda rope, timesteps: rope(timesteps))(rope, timesteps)

    table = _full_table(rope)
    _assert_close(result=embeddings.sines, reference=_select_reference(table.sines, timesteps))
    _assert_close(result=embeddings.cosines, reference=_select_reference(table.cosines, timesteps))


def test_rope_call_vmapped_over_batches_preserves_batch_sharding(fake_mesh: Mesh) -> None:
    rope = _rope()
    timesteps = jnp.array(
        [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
        ],
        dtype=jnp.int32,
    )
    timesteps = jax.device_put(timesteps, make_sharding((LogicalAxis.BATCH, None)))

    embeddings = call_vmapped(
        rope,
        timesteps,
        added_sharding_axis=make_test_sharding_config().resolve_axis(LogicalAxis.BATCH),
    )

    table = _full_table(rope)
    _assert_close(result=embeddings.sines, reference=_select_reference(table.sines, timesteps))
    _assert_close(result=embeddings.cosines, reference=_select_reference(table.cosines, timesteps))
    _assert_named_sharding(embeddings.sines.sharding, fake_mesh)
    _assert_named_sharding(embeddings.cosines.sharding, fake_mesh)
    assert embeddings.sines.sharding == make_sharding((LogicalAxis.BATCH, None, None))
    assert embeddings.cosines.sharding == make_sharding((LogicalAxis.BATCH, None, None))


def test_positional_embeddings_apply_matches_reference_and_preserves_sharding(fake_mesh: Mesh) -> None:
    rope = _rope()
    timesteps = jnp.array([0, 1, 3, 5], dtype=jnp.int32)
    embeddings = rope(timesteps)
    heads = _sharded_heads(jnp.arange(4 * 6, dtype=jnp.float32).reshape(4, 6) / 10)

    result = embeddings.apply(heads)

    _assert_close(result=result, reference=_apply_reference(embeddings, heads))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((LogicalAxis.BATCH, None))


def test_positional_embeddings_apply_rejects_too_small_head_dim() -> None:
    embeddings = _rope()(jnp.array([0, 1], dtype=jnp.int32))

    with pytest.raises(ValueError, match="exceeds input head_dim"):
        embeddings.apply(jnp.zeros((2, HEAD_DIM - 1), dtype=jnp.float32))


def test_rope_exports_no_arrays_and_regenerates_from_config() -> None:
    original = _rope()
    timesteps = jnp.array([0, 1, 3, 5], dtype=jnp.int32)

    assert original.export().arrays == {}

    template = original.config.init(EmptyInitializer(jnp.float32, make_test_sharding_config()))
    restored = template.load_exported(original.export())
    embeddings = restored(timesteps)

    table = _full_table(original)
    _assert_close(result=embeddings.sines, reference=_select_reference(table.sines, timesteps))
    _assert_close(result=embeddings.cosines, reference=_select_reference(table.cosines, timesteps))
