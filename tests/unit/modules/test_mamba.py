from math import prod

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from lalamo.initializer import RandomInitializer
from lalamo.modules.activations import SiLU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.token_mixer import MixerForwardPassConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.mamba import Mamba2, Mamba2Config
from tests.common import assert_close, tolerance

MODEL_DIM = 4
NUM_HEADS = 2
NUM_GROUPS = 1
HEAD_DIM = 2
STATE_DIM = 3
KERNEL_SIZE = 3
SEQUENCE_LENGTH = 10

SSM_CHUNK_CONFIGS = [
    pytest.param(2, 0, id="size-2-no-tail"),
    pytest.param(3, 0, id="size-3-chunk-tail"),
    pytest.param(4, 1, id="size-4-tail-threshold-1"),
    pytest.param(4, 3, id="size-4-recurrent-tail"),
    pytest.param(16, 16, id="all-recurrent"),
]


def _values(shape: tuple[int, ...], *, offset: int = 0, scale: float = 0.03) -> Array:
    return jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) * scale - 0.2


def _mamba() -> Mamba2:
    config = Mamba2Config(
        in_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        conv_config=SeparableCausalConvConfig(has_biases=False),
        activation=SiLU(),
        kernel_size=KERNEL_SIZE,
        num_heads=NUM_HEADS,
        num_groups=NUM_GROUPS,
        head_dim=HEAD_DIM,
        state_dim=STATE_DIM,
        expansion_factor=1,
        has_in_biases=False,
        has_out_biases=False,
    )
    return config.init(RandomInitializer(dtype=jnp.float32, key=jax.random.key(1)), model_dim=MODEL_DIM)


@pytest.mark.parametrize(("ssm_chunk_size", "ssm_min_tail_size_to_chunk"), SSM_CHUNK_CONFIGS)
@pytest.mark.parametrize("num_steps", [6, SEQUENCE_LENGTH], ids=["partial-prefix", "full-prefix"])
def test_mamba_chunked_scan_matches_recurrent_scan_for_ssm_chunk_config(
    ssm_chunk_size: int,
    ssm_min_tail_size_to_chunk: int,
    num_steps: int,
) -> None:
    module = _mamba()
    values = _values((SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM))
    keys = _values((SEQUENCE_LENGTH, NUM_GROUPS, STATE_DIM), offset=100)
    queries = _values((SEQUENCE_LENGTH, NUM_GROUPS, STATE_DIM), offset=200)
    dt = jax.nn.softplus(_values((SEQUENCE_LENGTH, NUM_HEADS), offset=300))
    initial_state = _values((NUM_HEADS, HEAD_DIM, STATE_DIM), offset=400)
    d = _values((NUM_HEADS,), offset=500)
    z = _values((SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM), offset=600)
    z_bias = _values((NUM_HEADS, HEAD_DIM), offset=700)
    forward_pass_config = MixerForwardPassConfig(
        ssm_chunk_size=ssm_chunk_size,
        ssm_min_tail_size_to_chunk=ssm_min_tail_size_to_chunk,
    )

    result_outputs, result_state = module._chunked_scan(  # noqa: SLF001
        values,
        keys,
        queries,
        dt,
        initial_state,
        forward_pass_config,
        num_steps,
        d=d,
        z=z,
        z_bias=z_bias,
    )
    reference_outputs, reference_state = module._recurrent_scan(  # noqa: SLF001
        values,
        keys,
        queries,
        dt,
        initial_state,
        num_steps,
        d=d,
        z=z,
        z_bias=z_bias,
    )

    with tolerance(atol=5e-4, rtol=5e-3):
        assert_close(result=result_outputs[:num_steps], reference=reference_outputs[:num_steps])
        assert_close(result=result_state, reference=reference_state)
