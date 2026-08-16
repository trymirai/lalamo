from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from lalamo.kernels.decode_attention import decode_attention
from lalamo.kernels.deltanet import deltanet_recurrent_scan
from lalamo.kernels.hadamard import hadamard_transform
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close, gpu_only

pytestmark = [gpu_only, pytest.mark.slow]


def _gpu_sharding_config(minimum_compute_capability: int) -> ShardingConfig:
    (device, *_) = jax.devices("gpu")
    if not device.device_kind.startswith("NVIDIA") or (
        float(getattr(device, "compute_capability", 0)) < minimum_compute_capability
    ):
        pytest.skip(f"requires an NVIDIA GPU with compute capability {minimum_compute_capability} or newer")
    return ShardingConfig.replicated((device,))


def _values(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> Array:
    return jax.random.normal(jax.random.key(seed), shape, dtype=jnp.float32) * scale


def _replicate(values: Array, sharding_config: ShardingConfig) -> Array:
    return jax.device_put(values, sharding_config.make_sharding((None,) * values.ndim))


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
    sharding_config = _gpu_sharding_config(10)
    queries = _replicate(
        _values((batch_size, 1, num_heads, head_dim), seed=batch_size, scale=0.2).astype(jnp.bfloat16),
        sharding_config,
    )
    keys = _replicate(
        _values((batch_size, capacity, num_groups, head_dim), seed=capacity, scale=0.1).astype(jnp.bfloat16),
        sharding_config,
    )
    values = _replicate(
        _values((batch_size, capacity, num_groups, head_dim), seed=capacity + 1, scale=0.1).astype(jnp.bfloat16),
        sharding_config,
    )
    lengths = capacity - batch_size + jnp.arange(batch_size)
    starts = 0 if window_size is None else jnp.maximum(lengths - window_size, 0)
    token_indices = jnp.arange(capacity)
    masks = (token_indices >= jnp.asarray(starts)[..., None]) & (token_indices < lengths[:, None])
    masks = masks.at[:, 8:12].set(False)
    masks = _replicate(masks[:, None], sharding_config)
    attention_scale = jnp.asarray(head_dim**-0.5 if scale is None else scale, dtype=jnp.float32)

    result = jax.jit(jax.vmap(decode_attention, in_axes=(0, 0, 0, None, 0, None, None)))(
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
    sharding_config = _gpu_sharding_config(9)
    queries = _replicate(_values((8, 1, 2, 128), seed=11, scale=0.1), sharding_config)
    keys = _replicate(_values((8, 1, 2, 128), seed=12, scale=0.1), sharding_config)
    values = _replicate(_values((8, 1, 2, 128), seed=13, scale=0.1), sharding_config)
    decay_factor = _replicate(-jax.nn.softplus(_values((8, 1, 2), seed=14)), sharding_config)
    beta = _replicate(jax.nn.sigmoid(_values((8, 1, 2), seed=15)), sharding_config)
    initial_state = _replicate(_values((8, 2, 128, 128), seed=16, scale=0.05), sharding_config)
    lengths = jnp.arange(8, dtype=jnp.int32) % 2
    lengths = _replicate(lengths, sharding_config)

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
    gpu_sharding_config = _gpu_sharding_config(9)
    values = _values((8, 512), seed=30, scale=0.1).astype(jnp.bfloat16)
    transform = jax.jit(jax.vmap(lambda row: hadamard_transform(row, block_size=128)))

    result = transform(_replicate(values, gpu_sharding_config))
    reference = transform(_replicate(values, cpu_sharding_config))

    assert_close(result=result, reference=reference, atol=2e-2, rtol=3e-2)
