from collections.abc import Iterator
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jaxtyping import DTypeLike

from lalamo.kernels.attention import pallas_decode_attention, xla_attention
from lalamo.kernels.attention.pallas_flash import triton_attention
from lalamo.utils.sharding import LogicalAxis, ShardingConfig
from tests.common import gpu_only


@pytest.fixture(autouse=True)
def attention_precision(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if jax.default_backend() == "cpu":
        monkeypatch.setattr(pl, "pallas_call", partial(pl.pallas_call, interpret=True))
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.fast
@pytest.mark.filterwarnings("error:.*falling back.*:RuntimeWarning")
@pytest.mark.parametrize(
    ("head_dim", "gqa", "query_count", "dtype", "has_bias", "cap"),
    [
        (64, 4, 1, jnp.bfloat16, True, None),
        (96, 8, 24, jnp.bfloat16, False, None),
        (128, 16, 8, jnp.bfloat16, True, 0.5),
        (256, 6, 32, jnp.bfloat16, False, None),
        (512, 1, 24, jnp.float16, True, None),
        (64, 3, 8, jnp.float32, True, 0.5),
        (96, 5, 24, jnp.float32, False, None),
        (128, 7, 32, jnp.float16, False, 0.5),
    ],
)
def test_attention_matches_reference_with_masks_bias_and_nonpower_geometry(
    head_dim: int, gqa: int, query_count: int, dtype: DTypeLike, has_bias: bool, cap: float | None
) -> None:
    heads, groups, capacity = 2 * gqa, 2, 37
    queries = (jax.random.normal(jax.random.key(1), (query_count, heads, head_dim)) * 0.2).astype(dtype)
    keys = (jax.random.normal(jax.random.key(2), (capacity, groups, head_dim)) * 0.2).astype(dtype)
    values = jax.random.normal(jax.random.key(3), keys.shape).astype(dtype)
    positions = jnp.arange(capacity)[None]
    mask = (positions < jnp.arange(query_count)[:, None] + 5) & (positions % 7 != 0)
    mask = mask.at[0].set(False)
    bias = None
    if has_bias:
        keys = keys.at[0].set(0)
        values = values.at[0].set(0)
        mask = mask.at[:, 0].set(True)
        bias = jnp.zeros((heads, query_count, capacity), dtype=dtype)
        bias = bias.at[:, :, 0].set(jnp.linspace(-2.0, 2.0, heads).astype(dtype)[:, None])
    attention = pallas_decode_attention
    if jax.default_backend() == "cpu":
        attention = partial(triton_attention, batch_size=1)
    actual = jax.jit(attention)(queries, keys, values, bias, mask, jnp.asarray(0.25), cap)
    expected = xla_attention(queries, keys, values, bias, mask, jnp.asarray(0.25), cap)
    tolerance = 0.025
    if dtype == jnp.float32:
        tolerance = 0.0002
    np.testing.assert_allclose(
        actual.astype(jnp.float32), expected.astype(jnp.float32), atol=tolerance, rtol=tolerance
    )


@gpu_only
@pytest.mark.slow
@pytest.mark.filterwarnings("error:.*falling back.*:RuntimeWarning")
@pytest.mark.parametrize(
    ("batch_size", "query_count", "head_dim", "dtype"),
    [
        (batch_size, query_count, head_dim, jnp.bfloat16)
        for head_dim in (64, 128, 256)
        for query_count in (1, 8, 24, 32, 512)
        for batch_size in (1, 2, 3, 4, 8, 16, 32, 64, 128)
    ]
    + [(512, 8, 64, jnp.bfloat16), (1_024, 1, 256, jnp.bfloat16)]
    + [
        (batch_size, query_count, head_dim, jnp.float32)
        for head_dim in (64, 128, 256)
        for query_count in (1, 8, 512)
        for batch_size in (1, 32)
    ],
)
def test_native_attention_batches_and_prefill_match_reference(
    batch_size: int, query_count: int, head_dim: int, dtype: DTypeLike
) -> None:
    heads, groups, capacity = 8, 2, max(64, query_count) + 7
    queries = (jax.random.normal(jax.random.key(1), (batch_size, query_count, heads, head_dim)) * 0.2).astype(dtype)
    keys = (jax.random.normal(jax.random.key(2), (batch_size, capacity, groups, head_dim)) * 0.2).astype(dtype)
    values = jax.random.normal(jax.random.key(3), keys.shape).astype(dtype)
    positions = jnp.arange(capacity)[None]
    mask = positions < capacity - query_count + jnp.arange(query_count)[:, None] + 1
    if head_dim in (128, 512):
        mask &= positions >= jnp.maximum(0, capacity - query_count + jnp.arange(query_count) - 31)[:, None]
    if query_count > 1:
        mask = mask.at[0].set(False)
    bias = None
    if head_dim == 64:
        keys = keys.at[:, 0].set(0)
        values = values.at[:, 0].set(0)
        mask = mask.at[:, 0].set(True)
        bias = jnp.zeros((heads, query_count, capacity), dtype=dtype)
        bias = bias.at[:, :, 0].set(jnp.linspace(-2.0, 2.0, heads).astype(dtype)[:, None])
    scales = jnp.linspace(0.1, 0.3, batch_size)
    caps = jnp.linspace(0.5, 1.5, batch_size)
    in_axes = (0, 0, 0, None, None, 0, 0)
    args = queries, keys, values, bias, mask, scales, caps
    actual = jax.jit(jax.vmap(pallas_decode_attention, in_axes=in_axes))(*args)
    expected = jax.jit(jax.vmap(xla_attention, in_axes=in_axes))(*args)
    tolerance = 0.025
    if dtype == jnp.float32:
        tolerance = 0.0002
    np.testing.assert_allclose(
        actual.astype(jnp.float32), expected.astype(jnp.float32), atol=tolerance, rtol=tolerance
    )


@pytest.mark.fast
@pytest.mark.filterwarnings("error:.*falling back.*:RuntimeWarning")
@pytest.mark.parametrize("device_count", [1, 2])
@pytest.mark.parametrize("partition_axis", [LogicalAxis.BATCH, LogicalAxis.MATRIX])
@pytest.mark.parametrize(
    ("batch_size", "query_count", "heads", "groups", "head_dim", "capacity", "dtype"),
    [(2, 8, 6, 2, 96, 37, jnp.float32), (32, 1, 24, 4, 256, 257, jnp.bfloat16)],
)
def test_dense_attention_preserves_explicit_sharding(
    device_count: int,
    partition_axis: LogicalAxis,
    batch_size: int,
    query_count: int,
    heads: int,
    groups: int,
    head_dim: int,
    capacity: int,
    dtype: DTypeLike,
) -> None:
    if len(jax.devices()) < device_count:
        pytest.skip("Requires two devices to exercise batch-sharded attention.")
    sharding = ShardingConfig.data_parallel(jax.devices()[:device_count])
    if partition_axis == LogicalAxis.MATRIX:
        sharding = ShardingConfig.tensor_parallel(jax.devices()[:device_count])
    partition = sharding.resolve_axis(partition_axis)
    query_axes = (partition, None, None, None)
    mask_axes = (partition, None, None)
    if partition_axis == LogicalAxis.MATRIX:
        query_axes = (None, None, partition, None)
        mask_axes = (None, None, None)
    queries = jax.device_put(
        jnp.zeros((batch_size, query_count, heads, head_dim), dtype=dtype),
        sharding.make_sharding(query_axes),
    )
    keys = jax.device_put(
        jnp.zeros((batch_size, capacity, groups, head_dim), dtype=dtype),
        sharding.make_sharding(query_axes),
    )
    row_values = jnp.arange(batch_size, dtype=jnp.float32)[:, None, None, None] / batch_size
    group_values = jnp.arange(groups, dtype=jnp.float32)[None, None, :, None] / (8 * groups)
    values = jax.device_put(jnp.broadcast_to(row_values + group_values, keys.shape).astype(dtype), keys.sharding)
    masks = jax.device_put(
        jnp.ones((batch_size, query_count, capacity), dtype=jnp.bool_), sharding.make_sharding(mask_axes)
    )
    attention = pallas_decode_attention
    if jax.default_backend() == "cpu":
        attention = partial(triton_attention, batch_size=batch_size)
    attention = jax.vmap(attention, in_axes=(0, 0, 0, None, 0, None, None))
    if jax.default_backend() == "cpu":
        attention = jax.shard_map(
            attention, mesh=sharding.mesh, out_specs=jax.typeof(queries).sharding.spec, check_vma=False
        )
    actual = jax.jit(attention)(queries, keys, values, None, masks, None, None)
    head_values = jnp.repeat(group_values, heads // groups, axis=2)
    np.testing.assert_allclose(actual, jnp.broadcast_to(row_values + head_values, queries.shape), atol=0.004)
    assert actual.sharding.is_equivalent_to(queries.sharding, actual.ndim)
