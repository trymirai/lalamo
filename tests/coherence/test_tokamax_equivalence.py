from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import Array

from lalamo.module import Keychain
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import (
    Normalization,
    NormalizationConfig,
    NormalizationForwardPassConfig,
    NormalizationImplementation,
    UpcastMode,
)
from lalamo.modules.token_mixer import AttentionImplementation, MixerForwardPassConfig
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from lalamo.weight_matrix import FullPrecisionSpec
from tests.common import assert_close
from tests.helpers import make_test_sharding_config

MODEL_DIM = 4
NUM_HEADS = 2
NUM_GROUPS = 2
HEAD_DIM = 2


def _weights(shape: tuple[int, ...], *, offset: int = 0) -> jax.Array:
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 20) - 0.5


def _linear(weights: Array, output_dims: tuple[int, ...]) -> Linear:
    return Linear(
        config=LinearConfig(),
        sharding_config=make_test_sharding_config(),
        weights=FullPrecisionSpec().compress(weights, sharding_config=make_test_sharding_config()),
        biases=None,
        output_dims=output_dims,
    )


def _attention() -> Attention:
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
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=False,
            has_out_biases=False,
            gate_projection_config=None,
        ),
        sharding_config=make_test_sharding_config(),
        qkv_projection=_linear(_weights((3 * qkv_dim, MODEL_DIM)), (qkv_dim, qkv_dim, qkv_dim)),
        gate_projection=None,
        out_projection=_linear(_weights((MODEL_DIM, qkv_dim), offset=100), (MODEL_DIM,)),
        query_norm=None,
        key_norm=None,
        sinks=None,
        borrows_kv_cache=False,
    )


def _normalization() -> Normalization:
    return Normalization(
        config=NormalizationConfig(
            epsilon=1e-5,
            scale_offset=0.25,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True,
            has_biases=True,
        ),
        sharding_config=make_test_sharding_config(),
        scales=jnp.array([1.0, 1.5, 2.0, 2.5], dtype=jnp.float32),
        biases=jnp.array([-0.25, 0.0, 0.25, 0.5], dtype=jnp.float32),
    )


def test_tokamax_attention_matches_standard() -> None:
    module = _attention()
    inputs = jnp.arange(5 * MODEL_DIM, dtype=jnp.float32).reshape(5, MODEL_DIM) / 10

    standard = module(
        inputs,
        positional_embeddings=None,
        forward_pass_config=MixerForwardPassConfig(
            attention_implementation=AttentionImplementation.STANDARD,
        ),
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )
    tokamax = module(
        inputs,
        positional_embeddings=None,
        forward_pass_config=MixerForwardPassConfig(
            attention_implementation=AttentionImplementation.TOKAMAX,
        ),
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )

    assert_close(result=tokamax.outputs, reference=standard.outputs)


def test_tokamax_normalization_matches_standard() -> None:
    module = _normalization()
    inputs = jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)

    standard = module(
        inputs,
        forward_pass_config=NormalizationForwardPassConfig(
            implementation=NormalizationImplementation.JAX,
        ),
    )
    tokamax = module(
        inputs,
        forward_pass_config=NormalizationForwardPassConfig(
            implementation=NormalizationImplementation.TOKAMAX,
        ),
    )

    assert_close(result=tokamax, reference=standard)
