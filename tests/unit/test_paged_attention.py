from collections.abc import Iterator
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jaxtyping import DTypeLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.inference.continuous_batching import (
    BatchingSequence,
    CachedPrefix,
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    TokenEvent,
    _merge_prefill,
)
from lalamo.initializer import RandomInitializer
from lalamo.kernels.attention import paged_decode_attention, xla_attention
from lalamo.models import GenerationConfig
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import Keychain
from lalamo.modules.token_mixer import MixerForwardPassConfig, State
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.kv_cache import PagedKVCacheLayer, PagedKVCachePool, StaticKVCacheLayer
from lalamo.utils.sharding import LogicalAxis, ShardingConfig
from tests.helpers import build_tiny_attention_decoder_config


@pytest.fixture(autouse=True)
def interpret_pallas(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if jax.default_backend() == "cpu":
        monkeypatch.setattr(pl, "pallas_call", partial(pl.pallas_call, interpret=True))
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    ("window", "sinks_present", "cap", "heads", "groups", "head_dim"),
    [
        (None, False, None, 4, 2, 64),
        (None, True, None, 64, 8, 64),
        (1, True, None, 4, 2, 64),
        (31, True, None, 4, 2, 64),
        (32, False, None, 4, 2, 64),
        (33, True, 0.5, 4, 2, 64),
        (128, True, None, 64, 8, 64),
        (128, False, 0.5, 4, 2, 64),
        (None, False, None, 64, 8, 96),
        (128, False, None, 32, 2, 128),
        (None, False, None, 16, 4, 256),
        (None, False, None, 24, 4, 256),
    ],
)
def test_paged_attention_matches_dense_across_pages_and_windows(
    dtype: DTypeLike,
    window: int | None,
    sinks_present: bool,
    cap: float | None,
    heads: int,
    groups: int,
    head_dim: int,
) -> None:
    lengths = jnp.array([0, 1, 31, 32, 33, 127, 128, 129], dtype=jnp.int32)
    batch_size = len(lengths)
    queries = (jax.random.normal(jax.random.key(1), (batch_size, heads, head_dim)) * 8).astype(dtype)
    keys = jax.random.normal(jax.random.key(2), (groups, 17, 32, head_dim)).astype(dtype)
    values = jax.random.normal(jax.random.key(3), keys.shape).astype(dtype)
    tables = jnp.stack([jnp.roll(jnp.arange(16, dtype=jnp.int32)[::-1], row) for row in range(batch_size)])
    sinks = None
    if sinks_present:
        sinks = jnp.linspace(-20.0, 20.0, heads)
    actual = paged_decode_attention(
        queries,
        keys,
        values,
        tables,
        lengths,
        scale=head_dim**-0.5,
        logit_soft_cap=cap,
        sinks=sinks,
        sliding_window_size=window,
    )

    dense_keys, dense_values = PagedKVCachePool(keys, values).read_pages(tables)
    positions = jnp.arange(dense_keys.shape[1])[None, :]
    valid = positions < lengths[:, None]
    if window is not None:
        valid &= positions >= jnp.maximum(0, lengths - window)[:, None]
    bias = None
    if sinks is not None:
        zeros = jnp.zeros((batch_size, 1, groups, head_dim), dtype=dtype)
        dense_keys = jnp.concatenate((zeros, dense_keys), axis=1)
        dense_values = jnp.concatenate((zeros, dense_values), axis=1)
        valid = jnp.concatenate((jnp.ones((batch_size, 1), dtype=jnp.bool), valid), axis=1)
        bias = jnp.zeros((heads, 1, dense_keys.shape[1]), dtype=dtype).at[:, :, 0].set(sinks[:, None].astype(dtype))
    expected = jax.vmap(
        lambda query, key, value, mask: xla_attention(query[None], key, value, bias, mask[None], head_dim**-0.5, cap)[
            0
        ]
    )(queries, dense_keys, dense_values, valid)

    tolerance = 1e-4
    if dtype == jnp.bfloat16:
        tolerance = 0.025
    np.testing.assert_allclose(
        actual.astype(jnp.float32), expected.astype(jnp.float32), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("pages", [1, 2, 16])
def test_sink_enters_normalizer_once_across_partitions(pages: int) -> None:
    actual = paged_decode_attention(
        jnp.zeros((1, 2, 64)),
        jnp.zeros((1, pages, 32, 64)),
        jnp.ones((1, pages, 32, 64)),
        jnp.arange(pages, dtype=jnp.int32)[None],
        jnp.array([32], dtype=jnp.int32),
        scale=0.125,
        logit_soft_cap=None,
        sinks=jnp.full(2, jnp.log(32.0)),
    )

    np.testing.assert_allclose(actual, 0.5, atol=1e-6)


@pytest.mark.parametrize("partition_axis", [None, LogicalAxis.BATCH, LogicalAxis.MATRIX])
def test_paged_attention_preserves_explicit_sharding(partition_axis: LogicalAxis | None) -> None:
    if len(jax.devices()) < 2:
        pytest.skip("Requires two devices to exercise the explicit tensor mesh boundary.")
    sharding = ShardingConfig.tensor_parallel(jax.devices()[:2])
    query_axes = (None, None, None)
    key_axes = (None, None, None, None)
    table_axes = (None, None)
    length_axes = (None,)
    sink_axes = (None,)
    if partition_axis == LogicalAxis.BATCH:
        sharding = ShardingConfig.data_parallel(jax.devices()[:2])
        query_axes = ("data", None, None)
        table_axes, length_axes = ("data", None), ("data",)
    if partition_axis == LogicalAxis.MATRIX:
        query_axes = (None, "tensor", None)
        key_axes, sink_axes = ("tensor", None, None, None), ("tensor",)
    queries = jax.device_put(jnp.zeros((2, 4, 64)), sharding.make_sharding(query_axes))
    keys = jax.device_put(jnp.zeros((2, 2, 32, 64)), sharding.make_sharding(key_axes))
    group_values = jnp.array([1.0, 3.0])[:, None, None, None]
    values = jax.device_put(jnp.broadcast_to(group_values, keys.shape), keys.sharding)
    tables = jax.device_put(jnp.array([[1, 0], [0, 1]], dtype=jnp.int32), sharding.make_sharding(table_axes))
    lengths = jax.device_put(jnp.array([33, 1], dtype=jnp.int32), sharding.make_sharding(length_axes))
    sink_counts = jnp.array([33.0, 66.0, 33.0, 66.0])
    sinks = jax.device_put(jnp.log(sink_counts), sharding.make_sharding(sink_axes))
    actual = jax.jit(partial(paged_decode_attention, scale=0.125, logit_soft_cap=None))(
        queries, keys, values, tables, lengths, sinks=sinks
    )
    token_counts = np.asarray(lengths, dtype=np.float32)[:, None]
    expected = np.array([[1.0, 1.0, 3.0, 3.0]]) * token_counts / (token_counts + np.asarray(sink_counts))
    np.testing.assert_allclose(actual, np.broadcast_to(expected[..., None], queries.shape), atol=1e-6)
    assert actual.sharding.is_equivalent_to(queries.sharding, actual.ndim)


@pytest.mark.parametrize(("window", "has_sinks", "cap"), [(None, True, None), (2, True, None), (2, False, 0.5)])
def test_attention_paged_decode_matches_dense_continuation(
    window: int | None, has_sinks: bool, cap: float | None
) -> None:
    sharding = ShardingConfig.replicated(jax.devices()[:1])
    config = build_tiny_attention_decoder_config((None,)).transformer_config.layer_configs[0].mixer_config
    assert isinstance(config, AttentionConfig)
    config = replace(
        config,
        num_heads=4,
        num_groups=2,
        head_dim=64,
        has_sinks=has_sinks,
        sliding_window_size=window,
        logit_soft_cap=cap,
    )
    attention = config.init(
        RandomInitializer(default_dtype=jnp.float32, sharding_config=sharding, key=jax.random.key(4)), model_dim=4
    )
    if has_sinks:
        attention = replace(attention, sinks=jnp.array([-2.0, 0.0, 2.0, 8.0]))
    forward = MixerForwardPassConfig.for_tracer_tests()
    keychain = Keychain.init(0, sharding_config=sharding)
    inputs = jax.random.normal(jax.random.key(5), (34, 4))
    prefix = attention(
        inputs[:-1],
        None,
        state=attention.init_static_state(33, jnp.float32),
        return_updated_state=True,
        forward_pass_config=forward,
        keychain=keychain,
    )
    assert isinstance(prefix.state, StaticKVCacheLayer)
    real_keys, real_values = (
        jnp.pad(cache[int(has_sinks) :], ((0, 31), (0, 0), (0, 0)))[None]
        for cache in (prefix.state.keys, prefix.state.values)
    )
    tables = jnp.array([[1, 0]], dtype=jnp.int32)
    pool = PagedKVCachePool(jnp.zeros((2, 2, 32, 64)), jnp.zeros((2, 2, 32, 64))).write_pages(
        tables, real_keys, real_values
    )
    paged = attention.paged_decode(
        inputs[-1:][None],
        None,
        PagedKVCacheLayer(pool.keys, pool.values, tables, jnp.array([33], dtype=jnp.int32)),
        forward,
        keychain=keychain,
    )
    dense = attention(inputs, None, forward_pass_config=forward, keychain=keychain)

    np.testing.assert_allclose(paged.outputs[0, 0], dense.outputs[-1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("has_sinks", [False, True])
@pytest.mark.parametrize("length", [31, 32, 33])
def test_static_prefill_to_pages_preserves_tokens_at_page_boundaries(has_sinks: bool, length: int) -> None:
    sharding = ShardingConfig.replicated(jax.devices()[:1])
    keys = jnp.arange(1, length + 1, dtype=jnp.float32)[:, None, None]
    cache = StaticKVCacheLayer.init(has_sinks, length, 1, 1, jnp.float32, sharding).extend(keys, keys * 2)
    pages = (length + 1 + 31) // 32
    tables = jnp.arange(pages, dtype=jnp.int32)[::-1][None]
    pool = PagedKVCachePool(jnp.zeros((1, pages, 32, 1)), jnp.zeros((1, pages, 32, 1)))
    # The merge copies page-rounded staging, including padding after the last real token.
    padded = replace(
        cache,
        keys=jnp.pad(cache.keys, ((0, pages * 32 + int(has_sinks) - cache.capacity), (0, 0), (0, 0))),
        values=jnp.pad(cache.values, ((0, pages * 32 + int(has_sinks) - cache.capacity), (0, 0), (0, 0))),
    )
    merged, _ = _merge_prefill(
        State((pool,)),
        jnp.zeros((1, 1)),
        State((jax.tree.map(lambda leaf: leaf[None], padded),)),
        jnp.zeros((1, 1)),
        tables,
        jnp.array([0], dtype=jnp.int32),
    )
    assert isinstance(merged[0], PagedKVCachePool)
    appended = PagedKVCacheLayer(
        merged[0].keys, merged[0].values, tables, jnp.array([length], dtype=jnp.int32)
    ).append(jnp.array([[[length + 1.0]]]), jnp.array([[[2 * (length + 1.0)]]]))
    read_keys, read_values = appended.read_pages(tables)

    np.testing.assert_array_equal(read_keys[0, : length + 1, 0, 0], jnp.arange(1, length + 2))
    np.testing.assert_array_equal(read_values[0, : length + 1, 0, 0], 2 * jnp.arange(1, length + 2))


@pytest.mark.parametrize("has_sinks", [False, True])
@pytest.mark.parametrize("length", [31, 32, 33])
def test_mixed_cached_and_blank_prefixes_match_fresh_attention(has_sinks: bool, length: int) -> None:
    sharding = ShardingConfig.replicated(jax.devices()[:1])
    decoder_config = build_tiny_attention_decoder_config((None,))
    (layer,) = decoder_config.transformer_config.layer_configs
    assert isinstance(layer.mixer_config, AttentionConfig)
    decoder_config = replace(
        decoder_config,
        transformer_config=replace(
            decoder_config.transformer_config,
            layer_configs=(replace(layer, mixer_config=replace(layer.mixer_config, has_sinks=has_sinks)),),
        ),
    )
    model = LanguageModelConfig(
        token_codec_config=ChatCodecConfig("", None, "system", "user", "assistant", None, None),
        decoder_config=decoder_config,
        generation_config=GenerationConfig(),
    ).init(
        Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        RandomInitializer(default_dtype=jnp.float32, sharding_config=sharding, key=jax.random.key(7)),
    )
    attention = model.decoder.transformer.layers[0].mixer
    assert isinstance(attention.config, AttentionConfig)
    forward = MixerForwardPassConfig.for_tracer_tests()
    keychain = Keychain.init(0, sharding_config=sharding)
    inputs = jax.random.normal(jax.random.key(8), (length + 1, 8))
    prefix = attention(
        inputs[:-1],
        None,
        state=attention.init_static_state(64, jnp.bfloat16),
        return_updated_state=True,
        forward_pass_config=forward,
        keychain=keychain,
    )
    assert isinstance(prefix.state, StaticKVCacheLayer)
    tables = jnp.array([[1, 0]], dtype=jnp.int32)
    pool = PagedKVCachePool(jnp.zeros((2, 3, 32, 4)), jnp.zeros((2, 3, 32, 4))).write_pages(
        tables, prefix.state.keys[None, int(has_sinks) :], prefix.state.values[None, int(has_sinks) :]
    )
    # Exercise the pure prefix mapping without starting the GPU-only engine scheduler.
    engine = object.__new__(ContinuousBatchingEngine)
    engine.model = model
    engine.config = ContinuousBatchingConfig(total_pages=2)
    engine.total_pages = 2
    engine._state = State((pool,))  # noqa: SLF001
    received: list[TokenEvent] = []
    batch = [
        BatchingSequence(
            tuple(range(length + 1)),
            1,
            (),
            received.extend,
            GenerationConfig().default_policy(),
            jax.random.key(0),
            return_logprobs=False,
            cached_prefix=CachedPrefix(tuple(range(length)), [1, 0], (None,)),
        ),
        BatchingSequence(
            (1,), 1, (), received.extend, GenerationConfig().default_policy(), jax.random.key(1), return_logprobs=False
        ),
    ]
    restored = engine._prefix_state(batch, [length, 0], 64)[0]  # noqa: SLF001
    assert isinstance(restored, StaticKVCacheLayer)
    for row, fresh_inputs in enumerate((inputs, inputs[-1:])):
        continued = attention(
            inputs[-1:],
            None,
            state=jax.tree.map(lambda leaf, row=row: leaf[row], restored),
            forward_pass_config=forward,
            keychain=keychain,
        )
        fresh = attention(
            fresh_inputs,
            None,
            state=attention.init_static_state(64, jnp.bfloat16),
            forward_pass_config=forward,
            keychain=keychain,
        )
        np.testing.assert_allclose(continued.outputs[-1], fresh.outputs[-1], rtol=0.025, atol=0.005)
