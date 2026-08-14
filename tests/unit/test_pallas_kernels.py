from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from lalamo.compressed.hybrid import HybridSpec, IncoherenceProcessingMode
from lalamo.compressed.int import IntSpec
from lalamo.initializer import RandomInitializer
from lalamo.kernels.deltanet import deltanet_recurrent_scan
from lalamo.kernels.hadamard import hadamard_transform
from lalamo.module import Keychain
from lalamo.modules.linear import LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixer import AttentionImplementation, MixerForwardPassConfig
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNet, DeltaNetConfig
from lalamo.modules.token_mixers.kv_cache import StaticKVCacheLayer
from lalamo.modules.utils import call_vmapped
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
    ("bits", "group_size", "rht_block_size", "shape", "is_symmetric"),
    [
        pytest.param(4, 16, 128, (1_536, 2_560), False, id="w4-g16-rht128"),
        pytest.param(4, 32, 32, (1_536, 2_560), False, id="w4-g32-rht32-even-stride"),
        pytest.param(4, 32, 32, (64, 9_216), False, id="w4-g32-rht32-wide"),
        pytest.param(8, 32, 32, (64, 256), False, id="w8-g32-rht32"),
        pytest.param(8, 64, 64, (64, 256), False, id="w8-g64-rht64"),
        pytest.param(8, 128, 128, (64, 256), False, id="w8-g128-rht128"),
    ],
)
def test_int_hybrid_dot_matches_decompressed_weights(
    bits: Literal[4, 8],
    group_size: int,
    rht_block_size: Literal[32, 64, 128] | None,
    shape: tuple[int, int],
    is_symmetric: bool,
) -> None:
    sharding_config = _gpu_sharding_config(9)
    _, columns = shape
    weights = _replicate(_values(shape, seed=bits, scale=0.1).astype(jnp.bfloat16), sharding_config)
    vector = _replicate(_values((columns,), seed=group_size, scale=0.2).astype(jnp.bfloat16), sharding_config)
    matrix = HybridSpec(
        quantization_spec=IntSpec(
            bits=bits,
            group_size=group_size,
            is_symmetric=is_symmetric,
        ),
        adapter_spec=None,
        incoherence_block_size=rht_block_size,
        incoherence_processing_mode=IncoherenceProcessingMode.INPUT,
    ).compress(
        weights,
        key=jax.random.key(0),
        sharding_config=sharding_config,
        is_sharded=False,
    )

    def dot(input_vector: Array) -> Array:
        return matrix.dot(
            input_vector,
            keychain=Keychain.init(0, sharding_config=sharding_config),
        )

    result = dot(vector)
    reference = vector @ matrix.decompress().T

    assert_close(result=result, reference=reference, atol=5e-2, rtol=3e-2)


def _attention(
    num_heads: int,
    num_groups: int,
    head_dim: int,
    scale: float | None,
    sliding_window_size: int | None,
    sharding_config: ShardingConfig,
) -> Attention:
    return AttentionConfig(
        qkv_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        query_norm_config=None,
        key_norm_config=None,
        num_heads=num_heads,
        num_groups=num_groups,
        head_dim=head_dim,
        is_causal=True,
        scale=scale,
        sliding_window_size=sliding_window_size,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
        gate_projection_config=None,
    ).init(
        RandomInitializer(
            default_dtype=jnp.bfloat16,
            sharding_config=sharding_config,
            key=jax.random.key(0),
        ),
        model_dim=8,
    )


def _run_attention(
    module: Attention,
    inputs: Array,
    state: StaticKVCacheLayer,
    implementation: AttentionImplementation,
) -> Array:
    return call_vmapped(
        lambda values, cache, *, keychain: module(
            values,
            positional_embeddings=None,
            state=cache,
            length_without_padding=1,
            forward_pass_config=MixerForwardPassConfig(
                attention_implementation=implementation,
                attention_tile_size=128,
            ),
            keychain=keychain,
        ),
        inputs,
        state,
        keychain=Keychain.init(0, sharding_config=module.sharding_config),
    ).outputs


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "num_groups", "head_dim", "capacity", "scale", "sliding_window_size"),
    [
        pytest.param(8, 16, 4, 256, 2_056, None, None, id="qwen35-b8-split8"),
        pytest.param(32, 40, 8, 128, 264, None, None, id="qwen3-d128-gqa5"),
        pytest.param(16, 64, 4, 128, 1_032, None, None, id="qwen3-d128-gqa16-split4"),
        pytest.param(64, 16, 8, 256, 2_056, 1.0, 1_024, id="gemma4-d256-sliding"),
        pytest.param(64, 16, 2, 512, 1_032, 1.0, None, id="gemma4-d512-gqa8"),
    ],
)
def test_decode_attention_matches_stable_reduction(
    batch_size: int,
    num_heads: int,
    num_groups: int,
    head_dim: int,
    capacity: int,
    scale: float | None,
    sliding_window_size: int | None,
) -> None:
    sharding_config = _gpu_sharding_config(10)
    module = _attention(
        num_heads,
        num_groups,
        head_dim,
        scale,
        sliding_window_size,
        sharding_config,
    )
    inputs = _replicate(
        _values((batch_size, 1, 8), seed=batch_size, scale=0.2).astype(jnp.bfloat16),
        sharding_config,
    )
    state = StaticKVCacheLayer(
        has_sinks=False,
        keys=_replicate(
            _values((batch_size, capacity, num_groups, head_dim), seed=capacity, scale=0.1).astype(jnp.bfloat16),
            sharding_config,
        ),
        values=_replicate(
            _values((batch_size, capacity, num_groups, head_dim), seed=capacity + 1, scale=0.1).astype(jnp.bfloat16),
            sharding_config,
        ),
        current_length=_replicate(
            (capacity - batch_size - 1 + jnp.arange(batch_size)).astype(jnp.int32),
            sharding_config,
        ),
    )

    result = _run_attention(module, inputs, state, AttentionImplementation.PALLAS)
    reference = _run_attention(module, inputs, state, AttentionImplementation.STABLE_REDUCTION)

    assert_close(result=result, reference=reference, atol=2e-2, rtol=3e-2)


def _deltanet(sharding_config: ShardingConfig) -> DeltaNet:
    return DeltaNetConfig(
        in_proj_config=LinearConfig(),
        conv_config=SeparableCausalConvConfig(has_biases=False),
        out_proj_config=LinearConfig(),
        norm_config=NormalizationConfig(
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        ),
        num_heads=2,
        num_groups=2,
        head_dim=128,
        value_head_dim=128,
        kernel_size=4,
    ).init(
        RandomInitializer(
            default_dtype=jnp.bfloat16,
            sharding_config=sharding_config,
            key=jax.random.key(0),
        ),
        model_dim=8,
    )


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
    assert jnp.array_equal(jax.device_get(final_state[::2]), jax.device_get(initial_state[::2]))


def test_pallas_hadamard_matches_cpu_under_jit_and_vmap() -> None:
    cpu_sharding_config = ShardingConfig.replicated(jax.devices("cpu")[:1])
    gpu_sharding_config = _gpu_sharding_config(9)
    values = _values((8, 512), seed=30, scale=0.1).astype(jnp.bfloat16)
    transform = jax.jit(jax.vmap(lambda row: hadamard_transform(row, block_size=128)))

    result = transform(_replicate(values, gpu_sharding_config))
    reference = transform(_replicate(values, cpu_sharding_config))

    assert_close(result=result, reference=reference, atol=2e-2, rtol=3e-2)
