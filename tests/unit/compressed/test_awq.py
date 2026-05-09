from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, Sharding

from lalamo.compressed.awq import AWQMatrix, AWQMatrixForInference, AWQMatrixForTraining, AWQSpec
from lalamo.compressed.utils.yaqa import yaqa_round_weights
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, Layout, Preconditioner, WeightMatrixSpec
from tests.unit.compressed import test_common as compressed_common

type Bits = Literal[4, 8]

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 4)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 5) / 7


def _preconditioned_weights() -> jax.Array:
    return jnp.array(
        [
            [-0.15443718, 0.08470728, -0.13598049, -0.15503626],
            [1.2666674, 0.14829758, 2.1415603, 1.0026742],
            [-0.29033586, 0.3583448, -0.70792735, -0.24555527],
            [0.8855825, 0.7861191, 0.88892716, 0.54932535],
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


def _manual_awq_affine_parameters(
    stored_weights: jax.Array,
    *,
    bits: int,
    group_size: int,
) -> tuple[jax.Array, jax.Array]:
    *leading_dims, rows, cols = stored_weights.shape
    grouped = stored_weights.reshape(*leading_dims, rows, cols // group_size, group_size)
    min_values = jnp.min(grouped, axis=-1)
    max_values = jnp.max(grouped, axis=-1)
    scales = jnp.maximum((max_values - min_values) / ((2**bits) - 1), jnp.finfo(stored_weights.dtype).eps)
    zero_points = jnp.nan_to_num(-min_values, nan=0, posinf=jnp.finfo(stored_weights.dtype).max, neginf=0)
    return scales, zero_points


def _round_unsigned(values: jax.Array, *, bits: int) -> jax.Array:
    return jnp.round(jnp.clip(values, 0, (2**bits) - 1))


def _manual_awq_dequantize(
    stored_weights: jax.Array,
    scales: jax.Array,
    zero_points: jax.Array,
    *,
    bits: int,
    group_size: int,
) -> jax.Array:
    int_zero_points = _round_unsigned(zero_points / scales, bits=bits)
    rounded_zero_points = int_zero_points * scales
    expanded_scales = jnp.repeat(scales, group_size, axis=-1)
    expanded_rounded_zero_points = jnp.repeat(rounded_zero_points, group_size, axis=-1)
    expanded_int_zero_points = jnp.repeat(int_zero_points, group_size, axis=-1)
    int_weights = _round_unsigned((stored_weights + expanded_rounded_zero_points) / expanded_scales, bits=bits)
    return (int_weights - expanded_int_zero_points) * expanded_scales


def _compress_pair(layout: Layout) -> tuple[AWQMatrixForTraining, AWQMatrixForInference]:
    weights = _logical_weights()
    spec = AWQSpec(bits=4, group_size=2, layout=layout)
    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)
    assert isinstance(training, AWQMatrixForTraining)
    assert isinstance(inference, AWQMatrixForInference)
    return training, inference


def _put_on_sharding(matrix: AWQMatrix, sharding: Sharding) -> AWQMatrix:
    if isinstance(matrix, AWQMatrixForTraining):
        return AWQMatrixForTraining(
            spec=matrix.spec,
            weights=jax.device_put(matrix.weights, sharding),
            scales=jax.device_put(matrix.scales, sharding),
            zero_points=jax.device_put(matrix.zero_points, sharding),
        )
    assert isinstance(matrix, AWQMatrixForInference)
    return AWQMatrixForInference(
        spec=matrix.spec,
        packed_weights=jax.device_put(matrix.packed_weights, sharding),
        scales=jax.device_put(matrix.scales, sharding),
        packed_zero_points=jax.device_put(matrix.packed_zero_points, sharding),
    )


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("bits", [4, 8])
def test_awq_spec_roundtrips_json(layout: Layout, bits: Bits) -> None:
    spec = AWQSpec(bits=bits, group_size=2, layout=layout)

    restored = WeightMatrixSpec.from_json(spec.to_json())

    assert restored == spec


@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_awq_compress_selects_requested_implementation(implementation: CompressionImplementation) -> None:
    matrix = AWQSpec(bits=4, group_size=2).compress(_logical_weights(), implementation=implementation)

    if implementation == CompressionImplementation.TRAINING:
        assert isinstance(matrix, AWQMatrixForTraining)
    else:
        assert isinstance(matrix, AWQMatrixForInference)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("bits", [4, 8])
def test_awq_compress_and_decompress_match_manual_min_max_quantization(layout: Layout, bits: Bits) -> None:
    weights = _logical_weights()
    spec = AWQSpec(bits=bits, group_size=2, layout=layout)
    stored_weights = _stored_weights(layout, weights)
    expected_scales, expected_zero_points = _manual_awq_affine_parameters(stored_weights, bits=bits, group_size=2)
    expected_decompressed = _manual_awq_dequantize(
        stored_weights,
        expected_scales,
        expected_zero_points,
        bits=bits,
        group_size=2,
    )
    expected_decompressed = layout.to_output_input(expected_decompressed)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)

    assert isinstance(training, AWQMatrixForTraining)
    assert isinstance(inference, AWQMatrixForInference)
    assert inference.packed_weights.dtype == jnp.uint8
    assert inference.packed_zero_points.dtype == jnp.uint8
    assert inference.packed_weights.shape == compressed_common.expected_packed_shape(stored_weights.shape, bits)
    assert inference.packed_zero_points.shape == compressed_common.expected_packed_shape(expected_scales.shape, bits)
    compressed_common.assert_close_arrays(result=training.scales, reference=expected_scales)
    compressed_common.assert_close_arrays(result=training.zero_points, reference=expected_zero_points)
    compressed_common.assert_close_arrays(result=inference.scales, reference=expected_scales)
    compressed_common.assert_close_arrays(result=training.decompress(), reference=expected_decompressed)
    compressed_common.assert_close_arrays(result=inference.decompress(), reference=expected_decompressed)


def test_awq_compress_uses_yaqa_weights_when_preconditioned() -> None:
    weights = _preconditioned_weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = AWQSpec(bits=4, group_size=2)

    yaqa_weights = yaqa_round_weights(weights, preconditioner, spec)
    preconditioned = spec.compress(weights, preconditioner=preconditioner)
    expected = spec.compress(yaqa_weights)

    compressed_common.assert_close_arrays(result=preconditioned.decompress(), reference=expected.decompress())


def test_awq_compress_maps_batched_preconditioned_weights() -> None:
    weights = jnp.stack([_preconditioned_weights(), _logical_weights()])
    input_blocks = jnp.stack([_input_block(), _input_block() + 0.1 * jnp.identity(4, dtype=jnp.float32)])
    output_blocks = jnp.stack([_output_block(), _output_block() + 0.1 * jnp.identity(4, dtype=jnp.float32)])
    preconditioner = Preconditioner.init(input_block=input_blocks, output_block=output_blocks)
    spec = AWQSpec(bits=4, group_size=2)

    def compress_one(weights: jax.Array, preconditioner: Preconditioner) -> jax.Array:
        return spec.compress(weights, preconditioner=preconditioner).decompress()

    result = spec.compress(weights, preconditioner=preconditioner).decompress()
    reference = jax.vmap(compress_one)(weights, preconditioner)

    compressed_common.assert_close_arrays(result=result, reference=reference)


def test_awq_compress_rejects_group_size_that_does_not_divide_stored_last_axis() -> None:
    weights = jnp.ones((4, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        AWQSpec(bits=4, group_size=2).compress(weights)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_awq_shape_dtype_struct_compress_uses_layout_shape_and_partition(
    fake_mesh: Mesh,
    layout: Layout,
    implementation: CompressionImplementation,
) -> None:
    spec = AWQSpec(bits=4, group_size=2, layout=layout)
    matrix = spec.compress(
        dummy_array((4, 4), jnp.float32),
        implementation=implementation,
    )
    expected_stored_shape = layout.weight_shape((), output_dim=4, input_dim=4)
    expected_partition = layout.weight_partition(num_leading_dims=0)
    expected_sharding = make_sharding(expected_partition)
    expected_packed_zero_points_sharding = make_sharding((*expected_partition[:-1], None))
    assert expected_sharding is not None
    assert expected_packed_zero_points_sharding is not None

    assert matrix.shape == expected_stored_shape
    assert matrix.dtype == jnp.float32
    assert matrix.scales.shape == (*expected_stored_shape[:-1], expected_stored_shape[-1] // spec.group_size)
    assert matrix.scales.sharding == expected_sharding
    compressed_common.assert_named_sharding(matrix.scales.sharding, fake_mesh)
    if isinstance(matrix, AWQMatrixForTraining):
        assert matrix.weights.shape == expected_stored_shape
        assert matrix.zero_points.shape == matrix.scales.shape
        assert matrix.weights.sharding == expected_sharding
        assert matrix.zero_points.sharding == expected_sharding
    else:
        assert isinstance(matrix, AWQMatrixForInference)
        assert matrix.packed_weights.shape == compressed_common.expected_packed_shape(
            expected_stored_shape,
            spec.bits,
        )
        assert matrix.packed_zero_points.shape == compressed_common.expected_packed_shape(
            matrix.scales.shape,
            spec.bits,
        )
        assert matrix.packed_weights.sharding == expected_sharding
        assert matrix.packed_zero_points.sharding == expected_packed_zero_points_sharding
        compressed_common.assert_named_sharding(matrix.packed_zero_points.sharding, fake_mesh)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_awq_export_contains_packed_weights_affine_parameters_and_spec(layout: Layout) -> None:
    matrix = AWQSpec(bits=4, group_size=2, layout=layout).compress(_logical_weights())

    exported = matrix.export()

    assert set(exported.arrays) == {"weights", "scales", "zero_points"}
    assert exported.metadata == {"spec": matrix.spec.to_json()}
    assert exported.arrays["weights"].dtype == jnp.uint8
    assert exported.arrays["zero_points"].dtype == jnp.uint8
    assert exported.arrays["weights"].shape == compressed_common.expected_packed_shape(matrix.shape, matrix.spec.bits)
    assert exported.arrays["zero_points"].shape == compressed_common.expected_packed_shape(
        matrix.scales.shape,
        matrix.spec.bits,
    )
    compressed_common.assert_close_arrays(result=exported.arrays["scales"], reference=matrix.scales)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("saved_implementation", list(CompressionImplementation))
@pytest.mark.parametrize("template_implementation", list(CompressionImplementation))
def test_awq_export_load_roundtrips_and_preserves_template_sharding(
    fake_mesh: Mesh,
    layout: Layout,
    saved_implementation: CompressionImplementation,
    template_implementation: CompressionImplementation,
) -> None:
    weights = _logical_weights()
    spec = AWQSpec(bits=4, group_size=2, layout=layout)
    saved_sharding = make_sharding((None, None))
    assert saved_sharding is not None
    original = _put_on_sharding(
        spec.compress(weights, implementation=saved_implementation),
        saved_sharding,
    )
    template = spec.compress(
        dummy_array(weights.shape, weights.dtype),
        implementation=template_implementation,
    )

    restored = template.load_exported(original.export())

    assert restored.spec == spec
    assert isinstance(restored, type(template))
    compressed_common.assert_close_arrays(result=restored.decompress(), reference=original.decompress())
    compressed_common.assert_named_sharding(restored.scales.sharding, fake_mesh)
    assert restored.scales.sharding == template.scales.sharding
    assert restored.scales.sharding != saved_sharding
    if isinstance(restored, AWQMatrixForTraining) and isinstance(template, AWQMatrixForTraining):
        assert restored.weights.sharding == template.weights.sharding
        assert restored.zero_points.sharding == template.zero_points.sharding
    elif isinstance(restored, AWQMatrixForInference) and isinstance(template, AWQMatrixForInference):
        assert restored.packed_weights.sharding == template.packed_weights.sharding
        assert restored.packed_zero_points.sharding == template.packed_zero_points.sharding


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_awq_from_packed_parameters_overrides_input_sharding(
    layout: Layout,
    implementation: CompressionImplementation,
) -> None:
    spec = AWQSpec(bits=4, group_size=2, layout=layout)
    original = spec.compress(_logical_weights(), implementation=CompressionImplementation.INFERENCE)
    template = spec.compress(
        dummy_array(_logical_weights().shape, _logical_weights().dtype),
        implementation=implementation,
    )
    saved_sharding = make_sharding((None, None))
    assert saved_sharding is not None
    assert isinstance(original, AWQMatrixForInference)

    restored = spec.from_packed_parameters(
        packed_weights=jax.device_put(original.packed_weights, saved_sharding),
        scales=jax.device_put(original.scales, saved_sharding),
        packed_zero_points=jax.device_put(original.packed_zero_points, saved_sharding),
        implementation=implementation,
    )

    compressed_common.assert_close_arrays(result=restored.decompress(), reference=original.decompress())
    assert isinstance(restored, type(template))
    assert restored.scales.sharding == template.scales.sharding
    if isinstance(restored, AWQMatrixForTraining) and isinstance(template, AWQMatrixForTraining):
        assert restored.weights.sharding == template.weights.sharding
        assert restored.zero_points.sharding == template.zero_points.sharding
    elif isinstance(restored, AWQMatrixForInference) and isinstance(template, AWQMatrixForInference):
        assert restored.packed_weights.sharding == template.packed_weights.sharding
        assert restored.packed_zero_points.sharding == template.packed_zero_points.sharding


def test_awq_load_rejects_spec_mismatch() -> None:
    original = AWQSpec(bits=4, group_size=2, layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    template = AWQSpec(bits=8, group_size=2, layout=Layout.INPUT_OUTPUT).compress(
        dummy_array((4, 4), jnp.float32),
    )

    with pytest.raises(ValueError, match="spec mismatch"):
        template.load_exported(original.export())


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_awq_training_and_inference_decompressions_match(layout: Layout) -> None:
    training, inference = _compress_pair(layout)

    compressed_common.assert_close_arrays(result=training.decompress(), reference=inference.decompress())
