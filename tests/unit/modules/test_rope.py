import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from einops import rearrange
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.initializer import EmptyInitializer, RandomInitializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import (
    LinearScalingRoPEConfig,
    LlamaRoPEConfig,
    LongRoPEConfig,
    PositionalEmbeddings,
    RoPE,
    RoPEConfig,
    UnscaledRoPEConfig,
    YARNRoPEConfig,
)
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
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
        pytest.param(
            LongRoPEConfig(
                base=10_000.0,
                max_sequence_length=NUM_TIMESTEPS,
                head_dim=HEAD_DIM,
                short_factor=(1.0, 1.0),
                long_factor=(1.0, 2.5),
                original_context_length=NUM_TIMESTEPS // 2,
                scaling_factor=32.0,
            ),
            id="longrope",
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


def _attention() -> Attention:
    norm_config = NormalizationConfig(
        epsilon=1e-6,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    config = AttentionConfig(
        qkv_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        num_heads=2,
        num_groups=2,
        head_dim=HEAD_DIM,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
    )
    return config.init(
        RandomInitializer(
            default_dtype=jnp.float32, sharding_config=make_test_sharding_config(), key=jax.random.key(1)
        ),
        model_dim=2 * HEAD_DIM,
    )


def test_attention_project_key_value_heads_matches_cache_written_by_call(fake_mesh: Mesh) -> None:
    module = _attention()
    num_tokens = 4
    inputs = jax.device_put(
        jnp.arange(num_tokens * module.model_dim, dtype=jnp.float32).reshape(num_tokens, module.model_dim) / 10,
        make_sharding((None, None)),
    )
    positional_embeddings = _rope()(jnp.arange(num_tokens, dtype=jnp.int32))

    keys, values = module.project_key_value_heads(
        inputs,
        positional_embeddings,
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )
    result = module(
        inputs,
        positional_embeddings,
        return_updated_state=True,
        keychain=Keychain.init(1, sharding_config=make_test_sharding_config()),
    )

    assert result.state is not None
    _assert_close(result=keys, reference=result.state.keys)
    _assert_close(result=values, reference=result.state.values)
    _assert_named_sharding(keys.sharding, fake_mesh)
    _assert_named_sharding(values.sharding, fake_mesh)

    projected = jnp.einsum(
        "ti,oi->to",
        jnp.asarray(jax.device_get(inputs)),
        jnp.asarray(jax.device_get(module.qkv_projection.weights.decompress())),
    )
    *_, raw_values = jnp.split(projected, Linear.get_split_points(module.qkv_projection.output_dims), axis=-1)
    _assert_close(
        result=values,
        reference=rearrange(raw_values, "tokens (groups head_channels) -> tokens groups head_channels", groups=2),
    )


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


LONGROPE_BASE = 10_000.0
LONGROPE_SHORT_FACTOR = (1.0, 1.0)
LONGROPE_LONG_FACTOR = (1.0, 2.5)
LONGROPE_ORIGINAL_CONTEXT_LENGTH = 4
LONGROPE_SCALING_FACTOR = 32.0


def _longrope_config(max_sequence_length: int) -> LongRoPEConfig:
    return LongRoPEConfig(
        base=LONGROPE_BASE,
        max_sequence_length=max_sequence_length,
        head_dim=HEAD_DIM,
        short_factor=LONGROPE_SHORT_FACTOR,
        long_factor=LONGROPE_LONG_FACTOR,
        original_context_length=LONGROPE_ORIGINAL_CONTEXT_LENGTH,
        scaling_factor=LONGROPE_SCALING_FACTOR,
    )


@pytest.mark.parametrize(
    ("max_sequence_length", "expected_factor"),
    [
        pytest.param(LONGROPE_ORIGINAL_CONTEXT_LENGTH, LONGROPE_SHORT_FACTOR, id="within-original-context"),
        pytest.param(NUM_TIMESTEPS, LONGROPE_LONG_FACTOR, id="beyond-original-context"),
    ],
)
def test_longrope_matches_huggingface_reference(
    max_sequence_length: int,
    expected_factor: tuple[float, ...],
) -> None:
    config = _longrope_config(max_sequence_length)

    channel_indices = jnp.arange(0, HEAD_DIM, 2, dtype=jnp.float32)
    reference_inverse_frequencies = 1.0 / (
        jnp.asarray(expected_factor) * LONGROPE_BASE ** (channel_indices / HEAD_DIM)
    )
    reference_attention_scaling = math.sqrt(
        1.0 + math.log(LONGROPE_SCALING_FACTOR) / math.log(LONGROPE_ORIGINAL_CONTEXT_LENGTH)
    )

    timesteps = jnp.arange(max_sequence_length, dtype=jnp.int32)
    reference_embeddings = jnp.outer(timesteps.astype(jnp.float32), reference_inverse_frequencies)
    reference_embeddings = jnp.concatenate((reference_embeddings, reference_embeddings), axis=-1)

    embeddings = config.compute_positional_embeddings(timesteps)

    _assert_close(result=embeddings.cosines, reference=jnp.cos(reference_embeddings) * reference_attention_scaling)
    _assert_close(result=embeddings.sines, reference=jnp.sin(reference_embeddings) * reference_attention_scaling)
