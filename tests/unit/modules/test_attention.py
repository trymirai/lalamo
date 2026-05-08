from math import prod

import jax.numpy as jnp
import pytest
from jaxtyping import Array

from lalamo.module import Keychain
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.token_mixer import AttentionImplementation, MixerForwardPassConfig
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from lalamo.weight_matrix import FullPrecisionSpec
from tests.common import assert_close

MODEL_DIM = 4
NUM_HEADS = 2
NUM_GROUPS = 2
HEAD_DIM = 2


def _linear(weights: Array, output_dims: tuple[int, ...]) -> Linear:
    return Linear(
        config=LinearConfig(),
        weights=FullPrecisionSpec().compress(weights),
        biases=None,
        output_dims=output_dims,
    )


def _weights(shape: tuple[int, ...], *, offset: int = 0) -> Array:
    values = jnp.arange(offset, offset + prod(shape), dtype=jnp.float32)
    return values.reshape(shape) / 20 - 0.5


def _attention(*, logit_soft_cap: float | None = None, has_sinks: bool = False) -> Attention:
    qkv_dim = NUM_HEADS * HEAD_DIM
    return Attention(
        config=AttentionConfig(
            qkv_projection_config=LinearConfig(),
            out_projection_config=LinearConfig(),
            query_norm_config=None,
            key_norm_config=None,
            num_heads=NUM_HEADS,
            num_groups=NUM_GROUPS,
            head_dim=HEAD_DIM,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=logit_soft_cap,
            has_sinks=has_sinks,
            has_qkv_biases=False,
            has_out_biases=False,
            gate_projection_config=None,
            use_rope=False,
        ),
        qkv_projection=_linear(_weights((3 * qkv_dim, MODEL_DIM)), (qkv_dim, qkv_dim, qkv_dim)),
        gate_projection=None,
        out_projection=_linear(_weights((MODEL_DIM, qkv_dim), offset=100), (MODEL_DIM,)),
        query_norm=None,
        key_norm=None,
        sinks=jnp.array([-1.25, 1.0], dtype=jnp.float32) if has_sinks else None,
    )


def _inputs() -> Array:
    return jnp.arange(5 * MODEL_DIM, dtype=jnp.float32).reshape(5, MODEL_DIM) / 10


def _output(module: Attention, implementation: AttentionImplementation) -> Array:
    return module(
        _inputs(),
        positional_embeddings=None,
        forward_pass_config=MixerForwardPassConfig(
            attention_implementation=implementation,
            attention_tile_size=2,
        ),
        keychain=Keychain.init(0),
    ).outputs


@pytest.mark.parametrize("logit_soft_cap", (None, 0.5))
def test_stable_reduction_attention_matches_standard(logit_soft_cap: float | None) -> None:
    module = _attention(logit_soft_cap=logit_soft_cap)

    standard = _output(module, AttentionImplementation.STANDARD)
    stable_reduction = _output(module, AttentionImplementation.STABLE_REDUCTION)

    assert_close(result=stable_reduction, reference=standard)


def test_tokamax_attention_rejects_soft_capping_with_sinks() -> None:
    module = _attention(logit_soft_cap=0.5, has_sinks=True)

    with pytest.raises(RuntimeError, match="soft-capping with additive bias"):
        _output(module, AttentionImplementation.TOKAMAX)
