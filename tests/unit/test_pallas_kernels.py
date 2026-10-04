import warnings
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.kernels.attention import pallas_decode_attention
from lalamo.kernels.deltanet import deltanet_recurrent_scan
from lalamo.kernels.deltanet.xla import xla_recurrent_scan
from lalamo.kernels.hadamard import hadamard_transform
from lalamo.kernels.mosaic import supports_mosaic_gpu
from lalamo.utils.sharding import LogicalAxis, ShardingConfig
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
    ("batch_size", "num_tokens", "num_heads"),
    [(1, 1, 16), (1, 8, 16), (3, 24, 32), (8, 31, 48), (128, 8, 16), (512, 1, 48)],
)
@pytest.mark.parametrize(
    "partitioned_axis",
    [None, LogicalAxis.BATCH, LogicalAxis.MATRIX],
    ids=["replicated", "data-parallel", "head-parallel"],
)
def test_deltanet_recurrence_matches_cpu_reference(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    partitioned_axis: LogicalAxis | None,
) -> None:
    sharding_config = ShardingConfig.replicated()
    if partitioned_axis is not None:
        devices = jax.devices()
        partitioned_size = batch_size
        if partitioned_axis is LogicalAxis.MATRIX:
            partitioned_size = num_heads
        device_count = max(count for count in range(1, len(devices) + 1) if partitioned_size % count == 0)
        if partitioned_axis is LogicalAxis.BATCH:
            sharding_config = ShardingConfig.data_parallel(devices[:device_count])
        else:
            sharding_config = ShardingConfig.tensor_parallel(devices[:device_count])
    if not supports_mosaic_gpu(sharding_config.mesh, minimum_compute_capability=9):
        pytest.skip("requires Hopper Pallas support")
    batch_axis = sharding_config.resolve_axis(LogicalAxis.BATCH)
    head_axis = sharding_config.resolve_axis(LogicalAxis.MATRIX)
    shapes = (
        (batch_size, num_tokens, num_heads, 128),
        (batch_size, num_tokens, num_heads, 128),
        (batch_size, num_tokens, num_heads, 128),
        (batch_size, num_tokens, num_heads),
        (batch_size, num_tokens, num_heads),
        (batch_size, num_heads, 128, 128),
    )
    shardings = (
        ((batch_axis, None, head_axis, None),) * 3
        + ((batch_axis, None, head_axis),) * 2
        + ((batch_axis, head_axis, None, None),)
    )
    arguments = tuple(
        jax.device_put(
            jax.random.normal(jax.random.key(index), shape, dtype=jnp.float32) * 0.1,
            sharding_config.make_sharding(sharding),
        )
        for index, (shape, sharding) in enumerate(zip(shapes, shardings, strict=True))
    )
    queries, keys, values, decay, beta, initial_state = arguments
    decay = -jax.nn.softplus(decay)
    beta = jax.nn.sigmoid(beta)
    lengths = jnp.arange(batch_size, dtype=jnp.int32) % (num_tokens + 2) - 1
    lengths = lengths.at[-1].set(num_tokens)
    arguments = (
        queries,
        keys,
        values,
        decay,
        beta,
        initial_state,
        jax.device_put(lengths, sharding_config.make_sharding((batch_axis,))),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Pallas DeltaNet recurrence .*falling back to XLA recurrence")
        result = jax.jit(jax.vmap(deltanet_recurrent_scan))(*arguments)
    cpu = jax.devices("cpu")[0]
    # Host materialization removes the GPU mesh from JAX's CPU oracle abstract types.
    reference = jax.jit(jax.vmap(xla_recurrent_scan))(
        *(jax.device_put(np.asarray(argument), cpu) for argument in arguments),
    )
    for actual, expected in zip(result, reference, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    inactive = np.asarray(lengths) <= 0
    np.testing.assert_array_equal(np.asarray(result[1])[inactive], np.asarray(initial_state)[inactive])


@pytest.mark.parametrize("num_tokens", [1, 8, 24, 31])
def test_deltanet_unbatched_and_shared_state_match_cpu_reference(num_tokens: int) -> None:
    sharding = ShardingConfig.replicated()
    if not supports_mosaic_gpu(sharding.mesh, minimum_compute_capability=9):
        pytest.skip("requires Hopper Pallas support")
    shapes = ((num_tokens, 16, 128),) * 3 + ((num_tokens, 16),) * 2 + ((16, 128, 128),)
    arguments = tuple(
        jax.random.normal(jax.random.key(index), shape, dtype=jnp.float32) * 0.1 for index, shape in enumerate(shapes)
    )
    queries, keys, values, decay, beta, state = arguments
    arguments = (queries, keys, values, -jax.nn.softplus(decay), jax.nn.sigmoid(beta), state, jnp.asarray(num_tokens))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Pallas DeltaNet recurrence .*falling back to XLA recurrence")
        result = jax.jit(deltanet_recurrent_scan)(*arguments)
        shared = jax.jit(jax.vmap(deltanet_recurrent_scan, in_axes=(0, 0, 0, None, None, None, None)))(
            jnp.stack([queries] * 3),
            jnp.stack([keys] * 3),
            jnp.stack([values] * 3),
            *arguments[3:],
        )
    cpu = jax.devices("cpu")[0]
    reference = jax.jit(xla_recurrent_scan)(*(jax.device_put(argument, cpu) for argument in arguments))
    for actual, batched, expected in zip(result, shared, reference, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(batched, np.stack([np.asarray(expected)] * 3), atol=2e-6, rtol=2e-5)


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
