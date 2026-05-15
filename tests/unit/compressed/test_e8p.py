from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, Sharding

from lalamo.compressed.e8p import (
    E8PMatrixForInference,
    E8PMatrixForTraining,
    E8PSpec,
    _codebook,
    _e81b_codebook,
    _round_to_codebook,
    _round_to_dense_codebook,
)
from lalamo.compressed.utils.yaqa import yaqa_round_weights
from lalamo.module import Keychain, ShardingAxis
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, Layout
from tests.common import assert_close_arrays, assert_named_sharding

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 8, 16)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 17) / 11


def _preconditioned_weights() -> jax.Array:
    return jnp.array(
        [
            [1.6226422, 2.0252647, -0.43359444, -0.07861735, 0.1760909, -0.97208923, -0.49529874, 0.4943786],
            [0.6643493, -0.9501635, 2.1795304, -1.9551506, 0.35857072, 0.15779513, 1.2770847, 1.5104648],
            [0.970656, 0.59960806, 0.0247007, -1.9164772, -1.8593491, 1.728144, 0.04719035, 0.814128],
            [0.13277933, -0.2367999, -0.5239256, 0.645061, -0.84720063, 1.8442863, -1.1845374, 1.3835493],
            [1.4451338, 0.85641253, 2.2180758, 0.5231655, 0.34664667, -0.19733144, -1.0545889, 1.2780242],
            [0.14533767, 0.23105137, 0.00865314, 0.39224577, 0.24148679, -1.321963, -0.8454542, -0.32263887],
            [-0.44387585, 0.12383226, -0.3314878, -0.01389739, -1.705842, 1.0063623, 0.63064164, 0.5953004],
            [0.4434663, 1.6999714, -1.0879849, -0.9296485, -0.46483088, -0.995943, 0.77838576, -0.42584738],
        ],
        dtype=jnp.float32,
    )


def _input_block() -> jax.Array:
    factors = jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 17
    return factors @ factors.T / 8 + jnp.eye(8, dtype=jnp.float32)


def _output_block() -> jax.Array:
    factors = jnp.flip(jnp.arange(64, dtype=jnp.float32).reshape(8, 8), axis=0) / 19
    return factors @ factors.T / 8 + jnp.eye(8, dtype=jnp.float32)


def _stored_weights(layout: Layout, weights: jax.Array) -> jax.Array:
    if layout == Layout.INPUT_OUTPUT:
        return weights.swapaxes(-1, -2)
    return weights


def _put_on_sharding(matrix: E8PMatrixForInference, sharding: Sharding) -> E8PMatrixForInference:
    residual_codes = matrix.residual_codes
    if residual_codes is not None:
        residual_codes = jax.device_put(residual_codes, sharding)
    return E8PMatrixForInference(
        spec=matrix.spec,
        is_sharded=matrix.is_sharded,
        dtype_=matrix.dtype,
        codes=jax.device_put(matrix.codes, sharding),
        residual_codes=residual_codes,
        scale=matrix.scale,
    )


def test_e8p_codebooks_have_expected_sizes() -> None:
    assert _codebook(jnp.float32).shape == (2**16, 8)
    assert _e81b_codebook(jnp.float32).shape == (2**8, 8)


def test_e8p_codebook_entries_live_on_quarter_grid() -> None:
    codebook = _codebook(jnp.float32)

    assert bool(jnp.all(jnp.round(codebook * 4) == codebook * 4))


def test_e8p_codebook_rounding_matches_dense_search() -> None:
    vectors = jax.random.normal(jax.random.key(0), (3, 5, 8))

    dense_values, _dense_codes = _round_to_dense_codebook(vectors, _codebook(vectors.dtype))
    values, _codes = _round_to_codebook(vectors)

    assert_close_arrays(result=values, reference=dense_values)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_e8p_compress_training_and_inference_match(bits: Literal[2, 3, 4], layout: Layout) -> None:
    weights = _logical_weights()
    spec = E8PSpec(bits=bits, layout=layout)
    stored_weights = _stored_weights(layout, weights)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)

    assert isinstance(training, E8PMatrixForTraining)
    assert isinstance(inference, E8PMatrixForInference)
    assert inference.codes.dtype == jnp.uint16
    assert inference.codes.shape == (*stored_weights.shape[:-1], stored_weights.shape[-1] // 8)
    if bits == 2:
        assert inference.residual_codes is None
    elif bits == 3:
        assert inference.residual_codes is not None
        assert inference.residual_codes.dtype == jnp.uint8
    else:
        assert inference.residual_codes is not None
        assert inference.residual_codes.dtype == jnp.uint16
    assert_close_arrays(result=training.decompress(), reference=inference.decompress())


def test_e8p_compress_uses_yaqa_weights_when_preconditioned() -> None:
    weights = _preconditioned_weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = E8PSpec(bits=2)

    yaqa_weights = yaqa_round_weights(weights, preconditioner, spec)
    preconditioned = spec.compress(weights, preconditioner=preconditioner)
    expected = spec.compress(yaqa_weights)

    assert_close_arrays(result=preconditioned.decompress(), reference=expected.decompress())


def test_e8p_compress_rejects_stored_last_axis_that_is_not_eight_aligned() -> None:
    weights = jnp.ones((8, 10), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        E8PSpec().compress(weights)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_e8p_export_load_roundtrips_and_preserves_template_sharding(
    bits: Literal[2, 3, 4],
    fake_mesh: Mesh,
) -> None:
    weights = _logical_weights()
    spec = E8PSpec(bits=bits, layout=Layout.INPUT_OUTPUT)
    saved_sharding = make_sharding((ShardingAxis.TENSOR, None))
    assert saved_sharding is not None
    original = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)
    assert isinstance(original, E8PMatrixForInference)
    original = _put_on_sharding(original, saved_sharding)
    template = spec.compress(
        dummy_array(weights.shape, weights.dtype),
        implementation=CompressionImplementation.INFERENCE,
    )

    restored = template.load_exported(original.export())

    assert restored.spec == spec
    assert isinstance(restored, type(template))
    assert_close_arrays(result=restored.decompress(), reference=original.decompress())
    assert isinstance(restored, E8PMatrixForInference)
    assert isinstance(template, E8PMatrixForInference)
    assert_named_sharding(restored.codes.sharding, fake_mesh)
    assert restored.codes.sharding == template.codes.sharding
    assert restored.codes.sharding != saved_sharding


def test_e8p_lookup_embedding_matches_decompressed_row() -> None:
    weights = _logical_weights()
    spec = E8PSpec(bits=3, layout=Layout.INPUT_OUTPUT)
    matrix = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)
    assert isinstance(matrix, E8PMatrixForInference)

    result = matrix.lookup_embedding(3, keychain=Keychain.init(3))

    assert_close_arrays(result=result, reference=spec.layout.from_output_input(matrix.decompress())[3, :])
