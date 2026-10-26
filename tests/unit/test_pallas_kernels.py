import warnings
from enum import Enum
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.initializer import RandomInitializer
from lalamo.kernels.attention import (
    pallas_decode_attention,
    pallas_sparse_attention,
    pallas_top_k_blocks,
    xla_sparse_attention,
    xla_top_k_blocks,
)
from lalamo.kernels.deltanet import deltanet_recurrent_scan
from lalamo.kernels.hadamard import hadamard_transform
from lalamo.kernels.mosaic import supports_mosaic_gpu
from lalamo.module import Keychain
from lalamo.modules.linear import LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixer import AttentionImplementation, MixerForwardPassConfig
from lalamo.modules.token_mixers.sparse_attention import SparseAttentionConfig, SparseAttentionIndexerConfig
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close, gpu_only

pytestmark = [gpu_only, pytest.mark.slow]


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "num_groups", "head_dim", "capacity", "scale", "window_size"),
    [
        pytest.param(8, 16, 4, 256, 2_056, None, None, id="qwen35-b8-split8"),
        pytest.param(32, 40, 8, 128, 264, None, None, id="qwen3-d128-gqa5"),
        pytest.param(16, 64, 4, 128, 1_032, None, None, id="qwen3-d128-gqa16-split4"),
        pytest.param(64, 16, 8, 256, 2_056, 1.0, 1_024, id="gemma4-d256-sliding"),
        pytest.param(64, 16, 1, 512, 1_032, 1.0, None, id="d512-mqa"),
    ],
)
def test_decode_attention_matches_reference(
    batch_size: int,
    num_heads: int,
    num_groups: int,
    head_dim: int,
    capacity: int,
    scale: float | None,
    window_size: int | None,
) -> None:
    sharding_config = ShardingConfig.replicated()
    if not supports_mosaic_gpu(sharding_config.mesh, minimum_compute_capability=10):
        pytest.skip("requires Blackwell Pallas support")
    queries = jax.device_put(
        (
            jax.random.normal(
                jax.random.key(batch_size),
                (batch_size, 1, num_heads, head_dim),
                dtype=jnp.float32,
            )
            * 0.2
        ).astype(jnp.bfloat16),
        sharding_config.make_sharding((None, None, None, None)),
    )
    keys = jax.device_put(
        (
            jax.random.normal(
                jax.random.key(capacity),
                (batch_size, capacity, num_groups, head_dim),
                dtype=jnp.float32,
            )
            * 0.1
        ).astype(jnp.bfloat16),
        sharding_config.make_sharding((None, None, None, None)),
    )
    values = jax.device_put(
        (
            jax.random.normal(
                jax.random.key(capacity + 1),
                (batch_size, capacity, num_groups, head_dim),
                dtype=jnp.float32,
            )
            * 0.1
        ).astype(jnp.bfloat16),
        sharding_config.make_sharding((None, None, None, None)),
    )
    lengths = capacity - batch_size + jnp.arange(batch_size)
    starts = 0 if window_size is None else jnp.maximum(lengths - window_size, 0)
    token_indices = jnp.arange(capacity)
    masks = (token_indices >= jnp.asarray(starts)[..., None]) & (token_indices < lengths[:, None])
    masks = masks.at[:, 8:12].set(False)
    masks = jax.device_put(
        masks[:, None],
        sharding_config.make_sharding((None, None, None)),
    )
    attention_scale = jnp.asarray(head_dim**-0.5 if scale is None else scale, dtype=jnp.float32)

    result = jax.jit(jax.vmap(pallas_decode_attention, in_axes=(0, 0, 0, None, 0, None, None)))(
        queries,
        keys,
        values,
        None,
        masks,
        attention_scale,
        None,
    )
    with jax.numpy_dtype_promotion("standard"):
        reference = jax.vmap(
            lambda query, key, value, mask: jax.nn.dot_product_attention(
                query,
                key,
                value,
                mask=mask,
                scale=cast("float", attention_scale),
            ),
        )(queries, keys, values, masks)

    assert_close(result=result, reference=reference, atol=2e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("batch_size", "num_queries", "num_heads", "num_groups", "head_dim", "capacity", "num_rows"),
    [
        pytest.param(2, 1, 24, 2, 256, 4_096, 2_052, id="qsa-decode"),
        pytest.param(1, 64, 24, 2, 256, 4_096, 2_052, id="qsa-prefill"),
        pytest.param(3, 5, 8, 1, 128, 1_000, 300, id="mqa-d128"),
    ],
)
def test_sparse_attention_matches_xla(
    batch_size: int,
    num_queries: int,
    num_heads: int,
    num_groups: int,
    head_dim: int,
    capacity: int,
    num_rows: int,
) -> None:
    sharding_config = ShardingConfig.replicated()
    if not supports_mosaic_gpu(sharding_config.mesh, minimum_compute_capability=10):
        pytest.skip("requires Blackwell Pallas support")

    def random_bfloat16(seed: int, shape: tuple[int, ...]) -> jax.Array:
        array = jax.random.normal(jax.random.key(seed), shape, dtype=jnp.float32).astype(jnp.bfloat16)
        return jax.device_put(array, sharding_config.make_sharding((None,) * len(shape)))

    queries = random_bfloat16(0, (batch_size, num_queries, num_heads, head_dim))
    keys = random_bfloat16(1, (batch_size, capacity, num_groups, head_dim))
    values = random_bfloat16(2, (batch_size, capacity, num_groups, head_dim))
    row_keys = jax.random.split(jax.random.key(3), batch_size * num_queries)
    unique_rows = jax.vmap(lambda key: jax.random.choice(key, capacity, (num_rows,), replace=False))(row_keys)
    is_masked = jax.random.bernoulli(jax.random.key(4), 0.125, unique_rows.shape)
    rows = jnp.where(is_masked, capacity, unique_rows).astype(jnp.int32).reshape(batch_size, num_queries, num_rows)
    rows = jax.device_put(rows, sharding_config.make_sharding((None, None, None)))
    attention_scale = jnp.asarray(head_dim**-0.5, dtype=jnp.float32)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Pallas sparse attention")
        result = jax.jit(jax.vmap(pallas_sparse_attention, in_axes=(0, 0, 0, 0, None, None)))(
            queries,
            keys,
            values,
            rows,
            attention_scale,
            None,
        )
    reference = jax.vmap(xla_sparse_attention, in_axes=(0, 0, 0, 0, None, None))(
        queries,
        keys,
        values,
        rows,
        attention_scale,
        None,
    )
    assert_close(result=result, reference=reference, atol=2e-2, rtol=3e-2)


class ScoreDistribution(Enum):
    NORMAL = "normal"
    QUANTIZED = "quantized"
    ZEROS = "zeros"


@pytest.mark.parametrize(
    ("batch_size", "num_queries", "num_blocks", "k", "distribution"),
    [
        pytest.param(4, 1, 65_536, 512, ScoreDistribution.NORMAL, id="decode-two-stage"),
        pytest.param(4, 1, 50_001, 512, ScoreDistribution.QUANTIZED, id="decode-ties-unaligned"),
        pytest.param(1, 512, 8_192, 512, ScoreDistribution.NORMAL, id="prefill-single-stage"),
        pytest.param(2, 300, 4_100, 512, ScoreDistribution.ZEROS, id="prefill-all-ties"),
        pytest.param(4, 3, 20_000, 100, ScoreDistribution.QUANTIZED, id="small-k"),
    ],
)
def test_top_k_blocks_matches_xla(
    batch_size: int,
    num_queries: int,
    num_blocks: int,
    k: int,
    distribution: ScoreDistribution,
) -> None:
    sharding_config = ShardingConfig.replicated()
    if not supports_mosaic_gpu(sharding_config.mesh, minimum_compute_capability=10):
        pytest.skip("requires Blackwell Pallas support")
    shape = (batch_size, num_queries, num_blocks)
    match distribution:
        case ScoreDistribution.NORMAL:
            scores = jax.random.normal(jax.random.key(0), shape, dtype=jnp.float32)
        case ScoreDistribution.QUANTIZED:
            scores = jnp.floor(jax.random.uniform(jax.random.key(0), shape) * 8) / 4
        case ScoreDistribution.ZEROS:
            scores = jnp.zeros(shape, dtype=jnp.float32)
    num_visible_blocks = jax.random.randint(jax.random.key(1), (batch_size * num_queries,), 0, num_blocks + 1)
    num_visible_blocks = num_visible_blocks.at[:4].set(jnp.array([num_blocks, k, k + 1, 0]))
    scores = jax.device_put(scores, sharding_config.make_sharding((None, None, None)))
    num_visible_blocks = jax.device_put(
        num_visible_blocks.reshape(batch_size, num_queries).astype(jnp.int32),
        sharding_config.make_sharding((None, None)),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Pallas top-k")
        result = jax.jit(jax.vmap(partial(pallas_top_k_blocks, k=k)))(scores, num_visible_blocks)
    reference = jax.vmap(partial(xla_top_k_blocks, k=k))(scores, num_visible_blocks)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(reference))


def test_sparse_attention_module_pallas_matches_xla() -> None:
    sharding_config = ShardingConfig.replicated()
    if not supports_mosaic_gpu(sharding_config.mesh, minimum_compute_capability=10):
        pytest.skip("requires Blackwell Pallas support")
    batch_size, prefill_length, model_dim = 2, 3_000, 512
    norm_config = NormalizationConfig(
        epsilon=1e-6,
        scale_offset=1.0,
        upcast_mode=UpcastMode.FULL_LAYER,
        subtract_mean=False,
    )
    config = SparseAttentionConfig(
        qkvg_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        num_heads=24,
        num_groups=2,
        head_dim=256,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkvg_biases=False,
        has_out_biases=False,
        has_gate=True,
        indexer_config=SparseAttentionIndexerConfig(
            projection_config=LinearConfig(),
            query_norm_config=norm_config,
            key_norm_config=norm_config,
            num_heads=4,
            head_dim=128,
            compression_ratio=4,
            token_budget=2_048,
        ),
    )
    module = config.init(
        RandomInitializer(default_dtype=jnp.bfloat16, sharding_config=sharding_config, key=jax.random.key(0)),
        model_dim=model_dim,
    )
    rope_config = UnscaledRoPEConfig(base=10_000_000.0, max_sequence_length=4_096, head_dim=64)
    inputs = jax.random.normal(jax.random.key(1), (batch_size, prefill_length + 1, model_dim)).astype(jnp.bfloat16)
    keychain = Keychain.init(0, sharding_config=sharding_config)

    @partial(jax.jit, static_argnames="implementation")
    def prefill_then_decode(
        implementation_inputs: jax.Array,
        implementation: AttentionImplementation,
    ) -> tuple[jax.Array, jax.Array]:
        forward_pass_config = MixerForwardPassConfig(attention_implementation=implementation)

        def run_sequence(sequence: jax.Array) -> tuple[jax.Array, jax.Array]:
            state = module.init_static_state(capacity=4_096, dtype=jnp.bfloat16)
            prefill = module(
                sequence[:prefill_length],
                rope_config.compute_positional_embeddings(jnp.arange(prefill_length)),
                state=state,
                return_updated_state=True,
                forward_pass_config=forward_pass_config,
                keychain=keychain,
            )
            decode = module(
                sequence[prefill_length:],
                rope_config.compute_positional_embeddings(jnp.arange(prefill_length, prefill_length + 1)),
                state=prefill.state,
                forward_pass_config=forward_pass_config,
                keychain=keychain,
            )
            return prefill.outputs, decode.outputs

        return jax.vmap(run_sequence)(implementation_inputs)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Pallas sparse attention")
        warnings.filterwarnings("error", message="Pallas top-k")
        pallas_prefill, pallas_decode = prefill_then_decode(inputs, AttentionImplementation.PALLAS)
    xla_prefill, xla_decode = prefill_then_decode(inputs, AttentionImplementation.XLA)
    assert_close(result=pallas_prefill, reference=xla_prefill, atol=2e-2, rtol=3e-2)
    assert_close(result=pallas_decode, reference=xla_decode, atol=2e-2, rtol=3e-2)


def test_deltanet_single_token_update_matches_recurrence_for_active_and_inactive_rows() -> None:
    sharding_config = ShardingConfig.replicated()
    if not supports_mosaic_gpu(sharding_config.mesh, minimum_compute_capability=9):
        pytest.skip("requires Hopper Pallas support")
    replicated_tensor_sharding = sharding_config.make_sharding((None, None, None, None))
    replicated_sequence_sharding = sharding_config.make_sharding((None, None, None))
    queries = jax.device_put(
        jax.random.normal(jax.random.key(11), (8, 1, 2, 128), dtype=jnp.float32) * 0.1,
        replicated_tensor_sharding,
    )
    keys = jax.device_put(
        jax.random.normal(jax.random.key(12), (8, 1, 2, 128), dtype=jnp.float32) * 0.1,
        replicated_tensor_sharding,
    )
    values = jax.device_put(
        jax.random.normal(jax.random.key(13), (8, 1, 2, 128), dtype=jnp.float32) * 0.1,
        replicated_tensor_sharding,
    )
    decay_factor = jax.device_put(
        -jax.nn.softplus(jax.random.normal(jax.random.key(14), (8, 1, 2), dtype=jnp.float32)),
        replicated_sequence_sharding,
    )
    beta = jax.device_put(
        jax.nn.sigmoid(jax.random.normal(jax.random.key(15), (8, 1, 2), dtype=jnp.float32)),
        replicated_sequence_sharding,
    )
    initial_state = jax.device_put(
        jax.random.normal(jax.random.key(16), (8, 2, 128, 128), dtype=jnp.float32) * 0.05,
        replicated_tensor_sharding,
    )
    lengths = jnp.arange(8, dtype=jnp.int32) % 2
    lengths = jax.device_put(lengths, sharding_config.make_sharding((None,)))

    update = jax.jit(jax.vmap(deltanet_recurrent_scan))
    outputs, final_state = update(queries, keys, values, decay_factor, beta, initial_state, lengths)

    decay = jnp.exp(decay_factor[:, 0, :, None, None])
    decayed_state = initial_state * decay
    value_delta = values[:, 0] - jnp.sum(decayed_state * keys[:, 0, :, None, :], axis=-1)
    value_delta = value_delta * beta[:, 0, :, None]
    updated_state = decayed_state + value_delta[..., None] * keys[:, 0, :, None, :]
    reference_state = jnp.where(lengths[:, None, None, None] > 0, updated_state, initial_state)
    reference_outputs = jnp.einsum("bhk,bhvk->bhv", queries[:, 0], updated_state)[:, None]

    assert_close(result=outputs, reference=reference_outputs, atol=5e-2, rtol=1e-1)
    assert_close(result=final_state, reference=reference_state, atol=1e-3, rtol=3e-2)


def test_pallas_hadamard_matches_cpu_under_jit_and_vmap() -> None:
    cpu_sharding_config = ShardingConfig.replicated(jax.devices("cpu")[:1])
    gpu_sharding_config = ShardingConfig.replicated()
    if not supports_mosaic_gpu(gpu_sharding_config.mesh, minimum_compute_capability=9):
        pytest.skip("requires Hopper Pallas support")
    values = (jax.random.normal(jax.random.key(30), (8, 512), dtype=jnp.float32) * 0.1).astype(jnp.bfloat16)
    transform = jax.jit(jax.vmap(lambda row: hadamard_transform(row, block_size=128)))

    result = transform(
        jax.device_put(
            values,
            gpu_sharding_config.make_sharding((None, None)),
        )
    )
    reference = transform(
        jax.device_put(
            values,
            cpu_sharding_config.make_sharding((None, None)),
        )
    )

    assert_close(result=result, reference=reference, atol=2e-2, rtol=3e-2)
