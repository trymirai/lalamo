import warnings
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from lalamo.initializer import RandomInitializer
from lalamo.kernels.deltanet import deltanet_recurrent_scan
from lalamo.module import Keychain, LogicalAxis, ShardingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixer import MixerForwardPassConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNet, DeltaNetConfig
from lalamo.modules.token_mixers.ssm_state import SSMStateLayer
from tests.common import assert_close
from tests.helpers import make_test_sharding_config

MODEL_DIM = 4
NUM_HEADS = 2
NUM_GROUPS = 2
HEAD_DIM = 3
VALUE_HEAD_DIM = 2
KERNEL_SIZE = 3
SEQUENCE_LENGTH = 10

SSM_CHUNK_CONFIGS = [
    pytest.param(2, 0, id="size-2-no-tail"),
    pytest.param(3, 0, id="size-3-chunk-tail"),
    pytest.param(4, 1, id="size-4-tail-threshold-1"),
    pytest.param(4, 3, id="size-4-recurrent-tail"),
    pytest.param(16, 16, id="all-recurrent"),
]


def _values(shape: tuple[int, ...], *, offset: int = 0, scale: float = 0.05) -> Array:
    return jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) * scale - 0.25


def _deltanet(
    initializer: RandomInitializer | None = None,
    *,
    head_dim: int = HEAD_DIM,
    value_head_dim: int = VALUE_HEAD_DIM,
) -> DeltaNet:
    config = DeltaNetConfig(
        in_proj_config=LinearConfig(),
        conv_config=SeparableCausalConvConfig(has_biases=False),
        out_proj_config=LinearConfig(),
        norm_config=NormalizationConfig(
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        ),
        num_heads=NUM_HEADS,
        num_groups=NUM_GROUPS,
        head_dim=head_dim,
        value_head_dim=value_head_dim,
        kernel_size=KERNEL_SIZE,
    )
    if initializer is None:
        initializer = RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
            key=jax.random.key(0),
        )
    return config.init(initializer, model_dim=MODEL_DIM)


@pytest.mark.parametrize(("ssm_chunk_size", "ssm_min_tail_size_to_chunk"), SSM_CHUNK_CONFIGS)
@pytest.mark.parametrize("num_steps", [6, SEQUENCE_LENGTH], ids=["partial-prefix", "full-prefix"])
def test_deltanet_chunked_scan_matches_recurrent_scan_for_ssm_chunk_config(
    ssm_chunk_size: int,
    ssm_min_tail_size_to_chunk: int,
    num_steps: int,
) -> None:
    module = _deltanet()
    queries = _values((SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM))
    keys = _values((SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM), offset=100)
    values = _values((SEQUENCE_LENGTH, NUM_HEADS, VALUE_HEAD_DIM), offset=200)
    decay_factor = -jax.nn.softplus(_values((SEQUENCE_LENGTH, NUM_HEADS), offset=300))
    beta = jax.nn.sigmoid(_values((SEQUENCE_LENGTH, NUM_HEADS), offset=400))
    initial_state = _values((NUM_HEADS, VALUE_HEAD_DIM, HEAD_DIM), offset=500)
    forward_pass_config = MixerForwardPassConfig(
        ssm_chunk_size=ssm_chunk_size,
        ssm_min_tail_size_to_chunk=ssm_min_tail_size_to_chunk,
    )

    outputs, final_state = module._chunked_scan(  # noqa: SLF001
        queries,
        keys,
        values,
        decay_factor,
        beta,
        initial_state,
        num_steps,
        forward_pass_config,
    )
    reference_outputs, reference_state = deltanet_recurrent_scan(
        queries,
        keys,
        values,
        decay_factor,
        beta,
        initial_state,
        num_steps,
    )

    assert_close(result=outputs[:num_steps], reference=reference_outputs[:num_steps])
    assert_close(result=final_state, reference=reference_state)


@pytest.mark.gpu
def test_deltanet_vmapped_pallas_fallback_warns() -> None:
    values = jnp.ones((2, 1, 1, 3), dtype=jnp.float32)
    factors = jnp.ones((2, 1, 1), dtype=jnp.float32)
    states = jnp.ones((2, 1, 3, 3), dtype=jnp.float32)

    with pytest.warns(RuntimeWarning, match="Pallas DeltaNet recurrence .*falling back to XLA recurrence"):
        jax.vmap(deltanet_recurrent_scan, in_axes=(0, 0, 0, 0, 0, 0, None))(
            values,
            values,
            values,
            factors,
            factors,
            states,
            1,
        )


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("num_tokens", [24, 32, 40], ids=["partial-chunk", "full-chunk", "chunk-and-tail"])
def test_deltanet_masked_continuation_preserves_state(backend: str, num_tokens: int) -> None:
    (device,) = jax.devices(backend)[:1]
    sharding = ShardingConfig.replicated([device])
    initializer = RandomInitializer(default_dtype=jnp.bfloat16, sharding_config=sharding, key=jax.random.key(0))
    module = _deltanet(initializer, head_dim=128, value_head_dim=64)
    with jax.default_device(device):
        state = SSMStateLayer(
            conv_state=initializer.normal(0.1, (2, module.conv_dim)),
            ssm_state=(
                jnp.arange(NUM_HEADS * 64 * 128, dtype=jnp.float32).reshape(NUM_HEADS, 64, 128) / 65536 + 1.000123
            ),
        )
        result = module(
            jnp.ones((num_tokens, 4), dtype=jnp.bfloat16),
            positional_embeddings=None,
            state=state,
            return_updated_state=True,
            length_without_padding=0,
            keychain=Keychain.init(0, sharding_config=sharding),
        )
    assert result.state is not None
    np.testing.assert_array_equal(result.state.conv_state, state.conv_state)
    np.testing.assert_array_equal(result.state.ssm_state, state.ssm_state)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_deltanet_batched_prefill_without_state_matches_independent_rows(backend: str, num_tokens: int) -> None:
    devices = jax.devices(backend)[:2]
    device, *_ = devices
    sharding = ShardingConfig.data_parallel(devices)
    with jax.default_device(device):
        module = _deltanet(
            RandomInitializer(default_dtype=jnp.float32, sharding_config=sharding, key=jax.random.key(0)),
            head_dim=128,
            value_head_dim=128,
        )
        inputs = jax.device_put(
            jax.random.normal(jax.random.key(2), (4, num_tokens, MODEL_DIM), dtype=jnp.float32),
            sharding.resolve_sharding((LogicalAxis.BATCH, None, None)),
        )
        lengths = jax.device_put(
            jnp.asarray([0, 1, num_tokens, num_tokens], dtype=jnp.int32),
            sharding.resolve_sharding((LogicalAxis.BATCH,)),
        )
        keychain = Keychain.init(0, sharding_config=sharding)

        def prefill(inputs: Array, length: Array) -> tuple[Array, SSMStateLayer]:
            result = module(
                inputs,
                positional_embeddings=None,
                return_updated_state=True,
                length_without_padding=length,
                forward_pass_config=MixerForwardPassConfig.for_tracer_tests(),
                keychain=keychain,
            )
            assert result.state is not None
            return result.outputs, result.state

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Pallas DeltaNet recurrence .*falling back to XLA recurrence")
            outputs, state = jax.jit(jax.vmap(prefill))(inputs, lengths)
        independent = [
            prefill(
                jax.device_put(row, sharding.make_sharding((None, None))),
                jax.device_put(length, sharding.make_sharding(())),
            )
            for row, length in zip(np.asarray(inputs), np.asarray(lengths), strict=True)
        ]
    np.testing.assert_allclose(outputs, np.stack([row for row, _ in independent]), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(
        state.ssm_state,
        np.stack([row.ssm_state for _, row in independent]),
        atol=2e-6,
        rtol=2e-5,
    )
    first_state = np.asarray(state.ssm_state)[0]
    np.testing.assert_array_equal(first_state, np.zeros_like(first_state))
