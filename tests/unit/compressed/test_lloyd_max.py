from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, Sharding

from lalamo.compressed.lloyd_max import (
    LloydMaxMatrixForInference,
    LloydMaxMatrixForTraining,
    LloydMaxSpec,
    _bias_lut,
    _codebook,
    _master_weights_to_grouped_weight_indices,
    _master_weights_to_quantized_weights,
)
from lalamo.compressed.utils.packing import pack_uint_to_uint8, packed_last_axis_dim, unpack_uint8_to_uint
from lalamo.compressed.utils.rounding import round_to_sorted_lut_table
from lalamo.compressed.utils.yaqa import yaqa_round_weights
from lalamo.module import Keychain, ShardingAxis
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, GradientEstimator, Layout, MatmulConfig
from tests.common import assert_close_arrays, assert_named_sharding

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


def _put_on_sharding(matrix: LloydMaxMatrixForInference, sharding: Sharding) -> LloydMaxMatrixForInference:
    packed_bias_indices = None
    if matrix.packed_bias_indices is not None:
        packed_bias_indices = jax.device_put(matrix.packed_bias_indices, sharding)
    return LloydMaxMatrixForInference(
        spec=matrix.spec,
        is_sharded=matrix.is_sharded,
        dtype_=matrix.dtype,
        packed_weight_indices=jax.device_put(matrix.packed_weight_indices, sharding),
        packed_scales=jax.device_put(matrix.packed_scales, sharding),
        packed_bias_indices=packed_bias_indices,
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_lloyd_max_pack_uint_roundtrips_dense_bit_widths(bits: int) -> None:
    values = (jnp.arange(17, dtype=jnp.uint8) % (2**bits)).reshape(1, 17)
    *_, cols = values.shape

    packed = pack_uint_to_uint8(values, bits)
    unpacked = unpack_uint8_to_uint(packed, bits=bits, unpacked_last_axis_dim=cols)

    *_, packed_cols = packed.shape
    assert packed_cols == packed_last_axis_dim(cols, bits)
    assert jnp.array_equal(unpacked, values)


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_codebook_is_symmetric_without_exact_zero(bits: int) -> None:
    codebook = _codebook(bits, group_size=4, dtype=jnp.float32)

    assert not bool(jnp.any(codebook == 0))
    assert_close_arrays(result=codebook, reference=-jnp.flip(codebook))


def test_codebook_depends_on_group_size() -> None:
    group_4_codebook = _codebook(bits=4, group_size=4, dtype=jnp.float32)
    group_32_codebook = _codebook(bits=4, group_size=32, dtype=jnp.float32)

    assert not bool(jnp.allclose(group_4_codebook, group_32_codebook))


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_lloyd_max_grouped_weight_indices_match_dense_nearest_codebook(
    bits: int,
) -> None:
    weights = jnp.linspace(-2, 2, 32, dtype=jnp.float32).reshape(2, 16)
    scales = jnp.array([[0.5, 1.25, 2.0, 0.25], [1.0, 0.75, 1.5, 2.5]], dtype=jnp.float32)
    codebook = _codebook(bits, group_size=4, dtype=jnp.float32)
    grouped_weights = weights.reshape(2, 4, 4)
    normalized_weights = grouped_weights / scales[..., None]
    dense_distances = jnp.abs(normalized_weights[..., None] - codebook)
    expected = jnp.argmin(dense_distances, axis=-1).astype(jnp.uint8)

    result = _master_weights_to_grouped_weight_indices(
        weights,
        scales,
        master_biases=None,
        bits=bits,
        group_size=4,
    )

    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_lloyd_max_compress_training_and_inference_match(
    layout: Layout,
    bits: Literal[2, 3, 4, 6, 8],
) -> None:
    weights = _logical_weights()
    spec = LloydMaxSpec(bits=bits, group_size=4, layout=layout)
    stored_weights = _stored_weights(layout, weights)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)

    assert isinstance(training, LloydMaxMatrixForTraining)
    assert isinstance(inference, LloydMaxMatrixForInference)
    assert inference.packed_weight_indices.dtype == jnp.uint8
    assert training.master_scales.dtype == weights.dtype
    assert inference.packed_scales.dtype == jnp.float8_e4m3fn
    assert training.master_biases is None
    assert inference.packed_bias_indices is None
    *leading_dims, cols = stored_weights.shape
    assert inference.packed_weight_indices.shape == (
        *leading_dims,
        packed_last_axis_dim(cols, bits),
    )
    assert_close_arrays(result=training.decompress(), reference=inference.decompress())


@pytest.mark.parametrize("bias_bits", [2, 3, 4, 6, 8])
def test_lloyd_max_biases_pack_and_roundtrip(bias_bits: Literal[2, 3, 4, 6, 8]) -> None:
    weights = _logical_weights()
    spec = LloydMaxSpec(bits=4, group_size=4, bias_bits=bias_bits)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)

    assert isinstance(training, LloydMaxMatrixForTraining)
    assert isinstance(inference, LloydMaxMatrixForInference)
    assert training.master_biases is not None
    assert inference.packed_bias_indices is not None
    *_, packed_bias_groups = inference.packed_bias_indices.shape
    *_, groups = inference.packed_scales.shape
    assert packed_bias_groups == packed_last_axis_dim(groups, bias_bits)
    assert_close_arrays(result=training.decompress(), reference=inference.decompress())


def test_lloyd_max_training_keeps_full_precision_master_weights_until_forward_pass() -> None:
    weights = _logical_weights()
    spec = LloydMaxSpec(bits=4, group_size=4, bias_bits=4)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)

    assert isinstance(training, LloydMaxMatrixForTraining)
    assert_close_arrays(result=training.master_weights, reference=weights)
    assert not bool(jnp.allclose(training.master_weights, training.decompress()))


def test_lloyd_max_training_dot_supports_stochastic_rounding_and_master_weight_gradients() -> None:
    weights = _logical_weights()
    out_channels, in_channels = weights.shape
    vector = jnp.linspace(-1, 1, in_channels, dtype=weights.dtype)
    spec = LloydMaxSpec(bits=4, group_size=4, bias_bits=4)
    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    assert isinstance(training, LloydMaxMatrixForTraining)
    forward_pass_config = MatmulConfig.for_training(GradientEstimator.STOCHASTIC_ROUNDING)

    result = training.dot(vector, keychain=Keychain.init(17), forward_pass_config=forward_pass_config)

    assert training.master_biases is not None

    def loss(
        master_weights: jax.Array,
        scales: jax.Array,
        biases: jax.Array,
    ) -> jax.Array:
        matrix = LloydMaxMatrixForTraining(
            spec=training.spec,
            is_sharded=training.is_sharded,
            master_weights=master_weights,
            master_scales=scales,
            master_biases=biases,
        )
        return jnp.sum(matrix.dot(vector, keychain=Keychain.init(17), forward_pass_config=forward_pass_config))

    gradients = jax.grad(loss, argnums=(0, 1, 2))(
        training.master_weights,
        training.master_scales,
        training.master_biases,
    )

    assert result.shape == (out_channels,)
    for gradient in gradients:
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.any(gradient != 0))


def test_lloyd_max_forward_uses_selected_estimator_for_master_biases() -> None:
    bits = 4
    bias_bits = 2
    group_size = 4
    bias_table = _bias_lut(group_size=group_size, bias_bits=bias_bits, bits=bits, dtype=jnp.float32)
    lower_bias = bias_table[1]
    upper_bias = bias_table[2]
    master_biases = (lower_bias + 0.51 * (upper_bias - lower_bias))[None, None]
    keychain = Keychain.init(0)
    bias_keychain, _weight_keychain = keychain.split()
    deterministic_biases = round_to_sorted_lut_table(
        master_biases,
        bias_table,
        keychain=None,
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )
    stochastic_biases = round_to_sorted_lut_table(
        master_biases,
        bias_table,
        keychain=bias_keychain,
        gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING,
    )
    assert not bool(jnp.all(deterministic_biases == stochastic_biases))

    master_weights = jnp.zeros((1, group_size), dtype=jnp.float32)
    scale_values = jnp.ones((1, 1), dtype=jnp.float32)
    stochastic_result = _master_weights_to_quantized_weights(
        master_weights,
        scale_values,
        master_biases=master_biases,
        bits=bits,
        bias_bits=bias_bits,
        group_size=group_size,
        keychain=keychain,
        gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING,
    )
    deterministic_bias_result = _master_weights_to_quantized_weights(
        master_weights,
        scale_values,
        master_biases=deterministic_biases,
        bits=bits,
        bias_bits=bias_bits,
        group_size=group_size,
        keychain=keychain,
        gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING,
    )

    assert not bool(jnp.allclose(stochastic_result, deterministic_bias_result))


def test_lloyd_max_compress_uses_yaqa_weights_when_preconditioned() -> None:
    weights = _preconditioned_weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = LloydMaxSpec(bits=4, group_size=4)

    yaqa_weights = yaqa_round_weights(weights, preconditioner, spec)
    preconditioned = spec.compress(weights, preconditioner=preconditioner)
    expected = spec.compress(yaqa_weights)

    assert_close_arrays(result=preconditioned.decompress(), reference=expected.decompress())


def test_lloyd_max_compress_rejects_group_size_that_does_not_divide_stored_last_axis() -> None:
    weights = jnp.ones((4, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        LloydMaxSpec(bits=4, group_size=2).compress(weights)


def test_lloyd_max_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    weights = _logical_weights()
    spec = LloydMaxSpec(bits=3, group_size=4, layout=Layout.INPUT_OUTPUT)
    saved_sharding = make_sharding((ShardingAxis.TENSOR, None))
    assert saved_sharding is not None
    original = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)
    assert isinstance(original, LloydMaxMatrixForInference)
    original = _put_on_sharding(original, saved_sharding)
    template = spec.compress(
        dummy_array(weights.shape, weights.dtype),
        implementation=CompressionImplementation.INFERENCE,
    )

    restored = template.load_exported(original.export())

    assert restored.spec == spec
    assert isinstance(restored, type(template))
    assert isinstance(restored, LloydMaxMatrixForInference)
    assert isinstance(template, LloydMaxMatrixForInference)
    assert_close_arrays(result=restored.decompress(), reference=original.decompress())
    assert_named_sharding(restored.packed_scales.sharding, fake_mesh)
    assert restored.packed_scales.sharding == template.packed_scales.sharding
    assert restored.packed_scales.sharding != saved_sharding
    assert restored.packed_weight_indices.sharding == template.packed_weight_indices.sharding
