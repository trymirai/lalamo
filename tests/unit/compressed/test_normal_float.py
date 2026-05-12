from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, Sharding

from lalamo.compressed.normal_float import (
    NormalFloatMatrixForInference,
    NormalFloatMatrixForTraining,
    NormalFloatSpec,
    _normal_float_codebook,
    _normal_float_grouped_indices,
)
from lalamo.compressed.utils.packing import pack_uint_to_uint8, packed_last_axis_dim, unpack_uint8_to_uint
from lalamo.compressed.utils.yaqa import yaqa_round_weights
from lalamo.module import Keychain, ShardingAxis
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, GradientEstimator, Layout, MatmulConfig
from tests.common import assert_close_arrays, assert_named_sharding

type NormalFloatBits = Literal[2, 3, 4, 6, 8]

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 8)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 7) / 5


def _preconditioned_weights() -> jax.Array:
    return jnp.array(
        [
            [1.6226422, 2.0252647, -0.43359444, -0.07861735],
            [0.1760909, -0.97208923, -0.49529874, 0.4943786],
            [0.6643493, -0.9501635, 2.1795304, -1.9551506],
            [0.35857072, 0.15779513, 1.2770847, 1.5104648],
        ],
        dtype=jnp.float32,
    )


def _input_block() -> jax.Array:
    return jnp.array(
        [
            [1.0, 0.7, -0.2, 0.1],
            [0.7, 1.4, 0.3, -0.1],
            [-0.2, 0.3, 1.1, 0.5],
            [0.1, -0.1, 0.5, 1.3],
        ],
        dtype=jnp.float32,
    )


def _output_block() -> jax.Array:
    return jnp.array(
        [
            [1.2, -0.5, 0.2, 0.1],
            [-0.5, 1.5, 0.4, -0.2],
            [0.2, 0.4, 1.1, 0.3],
            [0.1, -0.2, 0.3, 1.4],
        ],
        dtype=jnp.float32,
    )


def _stored_weights(layout: Layout, weights: jax.Array) -> jax.Array:
    if layout == Layout.INPUT_OUTPUT:
        return weights.swapaxes(-1, -2)
    return weights


def _put_on_sharding(matrix: NormalFloatMatrixForInference, sharding: Sharding) -> NormalFloatMatrixForInference:
    return NormalFloatMatrixForInference(
        spec=matrix.spec,
        is_sharded=matrix.is_sharded,
        dtype_=matrix.dtype,
        packed_weights=jax.device_put(matrix.packed_weights, sharding),
        scales=jax.device_put(matrix.scales, sharding),
        global_scales=jax.device_put(matrix.global_scales, make_sharding((None, None))),
        packed_bias_indices=None
        if matrix.packed_bias_indices is None
        else jax.device_put(matrix.packed_bias_indices, sharding),
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_normal_float_pack_uint_roundtrips_dense_bit_widths(bits: NormalFloatBits) -> None:
    values = (jnp.arange(17, dtype=jnp.uint8) % (2**bits)).reshape(1, 17)

    packed = pack_uint_to_uint8(values, bits)
    unpacked = unpack_uint8_to_uint(packed, bits=bits, unpacked_last_axis_dim=values.shape[-1])

    assert packed.shape[-1] == packed_last_axis_dim(values.shape[-1], bits)
    assert jnp.array_equal(unpacked, values)


def test_normal_float_nf4_codebook_matches_canonical_preserve_zero_values() -> None:
    expected = jnp.array(
        [
            -1.0,
            -0.6961928,
            -0.52507305,
            -0.3949175,
            -0.28444138,
            -0.18477343,
            -0.09105004,
            0.0,
            0.0795803,
            0.1609302,
            0.2461123,
            0.33791524,
            0.44070983,
            0.562617,
            0.72295684,
            1.0,
        ],
        dtype=jnp.float32,
    )

    result = _normal_float_codebook(4, preserve_zero=True, dtype=jnp.float32)

    assert_close_arrays(result=result, reference=expected)


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_normal_float_symmetric_codebook_does_not_preserve_zero(bits: NormalFloatBits) -> None:
    codebook = _normal_float_codebook(bits, preserve_zero=False, dtype=jnp.float32)

    assert not bool(jnp.any(codebook == 0))
    assert_close_arrays(result=codebook, reference=-jnp.flip(codebook))


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("preserve_zero", [False, True])
def test_normal_float_grouped_indices_match_dense_nearest_codebook(
    bits: NormalFloatBits,
    preserve_zero: bool,
) -> None:
    weights = jnp.linspace(-2, 2, 32, dtype=jnp.float32).reshape(2, 16)
    scales = jnp.array([[0.5, 1.25, 2.0, 0.25], [1.0, 0.75, 1.5, 2.5]], dtype=jnp.float32)
    codebook = _normal_float_codebook(bits, preserve_zero=preserve_zero, dtype=jnp.float32)
    grouped_weights = weights.reshape(2, 4, 4)
    normalized_weights = grouped_weights / scales[..., None]
    dense_distances = jnp.abs(normalized_weights[..., None] - codebook)
    expected = jnp.argmin(dense_distances, axis=-1).astype(jnp.uint8)

    result = _normal_float_grouped_indices(
        weights,
        scales,
        global_scales=None,
        biases=None,
        bits=bits,
        group_size=4,
        preserve_zero=preserve_zero,
    )

    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("preserve_zero", [False, True])
def test_normal_float_compress_training_and_inference_match(
    layout: Layout,
    bits: NormalFloatBits,
    preserve_zero: bool,
) -> None:
    weights = _logical_weights()
    spec = NormalFloatSpec(bits=bits, group_size=4, preserve_zero=preserve_zero, layout=layout)
    stored_weights = _stored_weights(layout, weights)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)

    assert isinstance(training, NormalFloatMatrixForTraining)
    assert isinstance(inference, NormalFloatMatrixForInference)
    assert inference.packed_weights.dtype == jnp.uint8
    assert inference.scales.dtype == jnp.float8_e4m3fn
    assert inference.global_scales.shape == (*stored_weights.shape[:-2], 1, 1)
    assert training.bias_indices is None
    assert inference.packed_bias_indices is None
    assert inference.packed_weights.shape == (
        *stored_weights.shape[:-1],
        packed_last_axis_dim(stored_weights.shape[-1], bits),
    )
    assert_close_arrays(result=training.decompress(), reference=inference.decompress())


def test_normal_float_rvq_biases_pack_and_roundtrip() -> None:
    weights = _logical_weights()
    spec = NormalFloatSpec(bits=4, group_size=4, bias_bits=4)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)

    assert isinstance(training, NormalFloatMatrixForTraining)
    assert isinstance(inference, NormalFloatMatrixForInference)
    assert training.bias_indices is not None
    assert inference.packed_bias_indices is not None
    assert inference.packed_bias_indices.shape[-1] == packed_last_axis_dim(inference.scales.shape[-1], 4)
    assert_close_arrays(result=training.decompress(), reference=inference.decompress())


def test_normal_float_training_keeps_full_precision_master_weights_until_forward_pass() -> None:
    weights = _logical_weights()
    spec = NormalFloatSpec(bits=4, group_size=4, bias_bits=4)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)

    assert isinstance(training, NormalFloatMatrixForTraining)
    assert_close_arrays(result=training.weights, reference=weights)
    assert not bool(jnp.allclose(training.weights, training.decompress()))


def test_normal_float_training_dot_supports_stochastic_rounding_and_master_weight_gradients() -> None:
    weights = _logical_weights()
    vector = jnp.linspace(-1, 1, weights.shape[-1], dtype=weights.dtype)
    spec = NormalFloatSpec(bits=4, group_size=4, bias_bits=4)
    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    assert isinstance(training, NormalFloatMatrixForTraining)
    forward_pass_config = MatmulConfig.for_training(GradientEstimator.STOCHASTIC_ROUNDING)

    result = training.dot(vector, keychain=Keychain.init(17), forward_pass_config=forward_pass_config)

    def loss(master_weights: jax.Array) -> jax.Array:
        matrix = NormalFloatMatrixForTraining(
            spec=training.spec,
            is_sharded=training.is_sharded,
            dtype_=training.dtype,
            weights=master_weights,
            scales=training.scales,
            global_scales=training.global_scales,
            bias_indices=training.bias_indices,
        )
        return jnp.sum(matrix.dot(vector, keychain=Keychain.init(17), forward_pass_config=forward_pass_config))

    gradients = jax.grad(loss)(training.weights)

    assert result.shape == (weights.shape[0],)
    assert bool(jnp.all(jnp.isfinite(gradients)))
    assert bool(jnp.any(gradients != 0))


def test_normal_float_rvq_biases_require_preserve_zero_nf4_and_supported_group_size() -> None:
    with pytest.raises(ValueError, match="preserve-zero NF4"):
        NormalFloatSpec(bits=3, group_size=4, bias_bits=4)

    with pytest.raises(ValueError, match="preserve-zero NF4"):
        NormalFloatSpec(bits=4, group_size=4, bias_bits=4, preserve_zero=False)

    with pytest.raises(ValueError, match="group size 8"):
        NormalFloatSpec(bits=4, group_size=8, bias_bits=4)


def test_normal_float_compress_uses_yaqa_weights_when_preconditioned() -> None:
    weights = _preconditioned_weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = NormalFloatSpec(bits=4, group_size=4)

    yaqa_weights = yaqa_round_weights(weights, preconditioner, spec)
    preconditioned = spec.compress(weights, preconditioner=preconditioner)
    expected = spec.compress(yaqa_weights)

    assert_close_arrays(result=preconditioned.decompress(), reference=expected.decompress())


def test_normal_float_compress_rejects_group_size_that_does_not_divide_stored_last_axis() -> None:
    weights = jnp.ones((4, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        NormalFloatSpec(bits=4, group_size=2).compress(weights)


def test_normal_float_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    weights = _logical_weights()
    spec = NormalFloatSpec(bits=3, group_size=4, layout=Layout.INPUT_OUTPUT)
    saved_sharding = make_sharding((ShardingAxis.TENSOR, None))
    assert saved_sharding is not None
    original = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)
    assert isinstance(original, NormalFloatMatrixForInference)
    original = _put_on_sharding(original, saved_sharding)
    template = spec.compress(
        dummy_array(weights.shape, weights.dtype),
        implementation=CompressionImplementation.INFERENCE,
    )

    restored = template.load_exported(original.export())

    assert restored.spec == spec
    assert isinstance(restored, type(template))
    assert_close_arrays(result=restored.decompress(), reference=original.decompress())
    assert_named_sharding(restored.scales.sharding, fake_mesh)
    assert_named_sharding(restored.global_scales.sharding, fake_mesh)
    assert restored.scales.sharding == template.scales.sharding
    assert restored.global_scales.sharding == template.global_scales.sharding
    assert restored.scales.sharding != saved_sharding
    assert isinstance(restored, NormalFloatMatrixForInference)
    assert isinstance(template, NormalFloatMatrixForInference)
    assert restored.packed_weights.sharding == template.packed_weights.sharding
