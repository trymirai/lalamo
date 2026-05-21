from math import prod

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from lalamo.initializer import RandomInitializer
from lalamo.modules.linear import LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixer import MixerForwardPassConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNet, DeltaNetConfig
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


def _deltanet() -> DeltaNet:
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
        head_dim=HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        kernel_size=KERNEL_SIZE,
    )
    return config.init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
            key=jax.random.key(0),
        ),
        model_dim=MODEL_DIM,
    )


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

    result = module._chunked_scan(  # noqa: SLF001
        queries,
        keys,
        values,
        decay_factor,
        beta,
        initial_state,
        num_steps,
        forward_pass_config,
    )
    reference = module._recurrent_scan(  # noqa: SLF001
        queries,
        keys,
        values,
        decay_factor,
        beta,
        initial_state,
        num_steps,
    )

    assert_close(result=result.outputs[:num_steps], reference=reference.outputs[:num_steps])
    assert_close(result=result.final_state, reference=reference.final_state)
