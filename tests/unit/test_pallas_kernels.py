from typing import cast

import jax
import jax.numpy as jnp
import pytest

from lalamo.kernels.attention import pallas_decode_attention
from lalamo.kernels.deltanet import deltanet_recurrent_scan
from lalamo.kernels.hadamard import hadamard_transform
from lalamo.kernels.mosaic import supports_mosaic_gpu
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
