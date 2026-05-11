from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding

from lalamo.compressed.mlx import MLXMatrix, MLXMatrixForInference, MLXMatrixForTraining, MLXSpec
from lalamo.compressed.utils.yaqa import yaqa_round_weights
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, Layout, WeightMatrixSpec
from tests.common import assert_close

type Bits = Literal[4, 8]

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 4)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 3) / 5


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


def _manual_mlx_affine_parameters(
    stored_weights: jax.Array,
    *,
    bits: int,
    group_size: int,
) -> tuple[jax.Array, jax.Array]:
    *leading_dims, rows, cols = stored_weights.shape
    grouped = stored_weights.reshape(*leading_dims, rows, cols // group_size, group_size)
    biases = jnp.min(grouped, axis=-1)
    max_values = jnp.max(grouped, axis=-1)
    scales = jnp.maximum((max_values - biases) / ((2**bits) - 1), jnp.finfo(stored_weights.dtype).eps)
    return scales, biases


def _manual_mlx_dequantize(
    stored_weights: jax.Array,
    scales: jax.Array,
    biases: jax.Array,
    *,
    bits: int,
    group_size: int,
) -> jax.Array:
    expanded_scales = jnp.repeat(scales, group_size, axis=-1)
    expanded_biases = jnp.repeat(biases, group_size, axis=-1)
    quantized = jnp.round(jnp.clip((stored_weights - expanded_biases) / expanded_scales, 0, (2**bits) - 1))
    return quantized * expanded_scales + expanded_biases


def _packed_last_dim(last_dim: int, bits: int) -> int:
    if bits == 8:
        return last_dim
    return last_dim // (8 // bits)


def _expected_packed_shape(stored_shape: tuple[int, ...], bits: int) -> tuple[int, ...]:
    *leading_dims, cols = stored_shape
    return (*leading_dims, _packed_last_dim(cols, bits))


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _put_on_sharding(matrix: MLXMatrix, sharding: Sharding) -> MLXMatrix:
    if isinstance(matrix, MLXMatrixForTraining):
        return MLXMatrixForTraining(
            spec=matrix.spec,
            is_sharded=matrix.is_sharded,
            weights=jax.device_put(matrix.weights, sharding),
            scales=jax.device_put(matrix.scales, sharding),
            biases=jax.device_put(matrix.biases, sharding),
        )
    assert isinstance(matrix, MLXMatrixForInference)
    return MLXMatrixForInference(
        spec=matrix.spec,
        is_sharded=matrix.is_sharded,
        packed_weights=jax.device_put(matrix.packed_weights, sharding),
        scales=jax.device_put(matrix.scales, sharding),
        biases=jax.device_put(matrix.biases, sharding),
    )


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("bits", [4, 8])
def test_mlx_spec_roundtrips_json(layout: Layout, bits: Bits) -> None:
    spec = MLXSpec(bits=bits, group_size=2, layout=layout)

    restored = WeightMatrixSpec.from_json(spec.to_json())

    assert restored == spec


@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_mlx_compress_selects_requested_implementation(implementation: CompressionImplementation) -> None:
    matrix = MLXSpec(bits=4, group_size=2).compress(_logical_weights(), implementation=implementation)

    if implementation == CompressionImplementation.TRAINING:
        assert isinstance(matrix, MLXMatrixForTraining)
    else:
        assert isinstance(matrix, MLXMatrixForInference)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("bits", [4, 8])
def test_mlx_compress_and_decompress_match_manual_min_max_quantization(layout: Layout, bits: Bits) -> None:
    weights = _logical_weights()
    spec = MLXSpec(bits=bits, group_size=2, layout=layout)
    stored_weights = _stored_weights(layout, weights)
    expected_scales, expected_biases = _manual_mlx_affine_parameters(stored_weights, bits=bits, group_size=2)
    expected_decompressed = _manual_mlx_dequantize(
        stored_weights,
        expected_scales,
        expected_biases,
        bits=bits,
        group_size=2,
    )
    expected_decompressed = layout.to_output_input(expected_decompressed)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)

    assert isinstance(training, MLXMatrixForTraining)
    assert isinstance(inference, MLXMatrixForInference)
    assert inference.packed_weights.dtype == jnp.uint8
    assert inference.packed_weights.shape == _expected_packed_shape(stored_weights.shape, bits)
    _assert_close(result=training.scales, reference=expected_scales)
    _assert_close(result=training.biases, reference=expected_biases)
    _assert_close(result=inference.scales, reference=expected_scales)
    _assert_close(result=inference.biases, reference=expected_biases)
    _assert_close(result=training.decompress(), reference=expected_decompressed)
    _assert_close(result=inference.decompress(), reference=expected_decompressed)


def test_mlx_compress_uses_yaqa_weights_when_preconditioned() -> None:
    weights = _preconditioned_weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = MLXSpec(bits=4, group_size=4)

    yaqa_weights = yaqa_round_weights(weights, preconditioner, spec)
    preconditioned = spec.compress(weights, preconditioner=preconditioner)
    expected = spec.compress(yaqa_weights)

    _assert_close(result=preconditioned.decompress(), reference=expected.decompress())


def test_mlx_compress_maps_batched_preconditioned_weights() -> None:
    weights = jnp.stack([_preconditioned_weights(), _logical_weights()])
    input_blocks = jnp.stack([_input_block(), _input_block() + 0.1 * jnp.identity(4, dtype=jnp.float32)])
    output_blocks = jnp.stack([_output_block(), _output_block() + 0.1 * jnp.identity(4, dtype=jnp.float32)])
    preconditioner = Preconditioner.init(input_block=input_blocks, output_block=output_blocks)
    spec = MLXSpec(bits=4, group_size=4)

    def compress_one(weights: jax.Array, preconditioner: Preconditioner) -> jax.Array:
        return spec.compress(weights, preconditioner=preconditioner).decompress()

    result = spec.compress(weights, preconditioner=preconditioner).decompress()
    reference = jax.vmap(compress_one)(weights, preconditioner)

    _assert_close(result=result, reference=reference)


def test_mlx_compress_rejects_group_size_that_does_not_divide_stored_last_axis() -> None:
    weights = jnp.ones((4, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        MLXSpec(bits=4, group_size=2).compress(weights)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_mlx_shape_dtype_struct_compress_uses_layout_shape_and_partition(
    fake_mesh: Mesh,
    layout: Layout,
    implementation: CompressionImplementation,
) -> None:
    spec = MLXSpec(bits=4, group_size=2, layout=layout)
    matrix = spec.compress(
        dummy_array((4, 4), jnp.float32),
        implementation=implementation,
    )
    expected_stored_shape = layout.weight_shape((), output_dim=4, input_dim=4)
    expected_sharding = make_sharding(layout.weight_partition(num_leading_dims=0))

    assert matrix.shape == expected_stored_shape
    assert matrix.dtype == jnp.float32
    assert matrix.scales.shape == (*expected_stored_shape[:-1], expected_stored_shape[-1] // spec.group_size)
    assert matrix.biases.shape == matrix.scales.shape
    assert matrix.scales.sharding == expected_sharding
    assert matrix.biases.sharding == expected_sharding
    _assert_named_sharding(matrix.scales.sharding, fake_mesh)
    if isinstance(matrix, MLXMatrixForTraining):
        assert matrix.weights.shape == expected_stored_shape
        assert matrix.weights.sharding == expected_sharding
    else:
        assert isinstance(matrix, MLXMatrixForInference)
        assert matrix.packed_weights.shape == _expected_packed_shape(expected_stored_shape, spec.bits)
        assert matrix.packed_weights.sharding == expected_sharding


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_mlx_export_contains_packed_weights_affine_parameters_and_spec(layout: Layout) -> None:
    matrix = MLXSpec(bits=4, group_size=2, layout=layout).compress(_logical_weights())

    exported = matrix.export()

    assert set(exported.arrays) == {"weights", "scales", "biases"}
    assert exported.metadata == {"spec": matrix.spec.to_json()}
    assert exported.arrays["weights"].dtype == jnp.uint8
    assert exported.arrays["weights"].shape == _expected_packed_shape(matrix.shape, matrix.spec.bits)
    _assert_close(result=exported.arrays["scales"], reference=matrix.scales)
    _assert_close(result=exported.arrays["biases"], reference=matrix.biases)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("saved_implementation", list(CompressionImplementation))
@pytest.mark.parametrize("template_implementation", list(CompressionImplementation))
def test_mlx_export_load_roundtrips_and_preserves_template_sharding(
    fake_mesh: Mesh,
    layout: Layout,
    saved_implementation: CompressionImplementation,
    template_implementation: CompressionImplementation,
) -> None:
    weights = _logical_weights()
    spec = MLXSpec(bits=4, group_size=2, layout=layout)
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
    _assert_close(result=restored.decompress(), reference=original.decompress())
    _assert_named_sharding(restored.scales.sharding, fake_mesh)
    assert restored.scales.sharding == template.scales.sharding
    assert restored.scales.sharding != saved_sharding
    assert restored.biases.sharding == template.biases.sharding
    if isinstance(restored, MLXMatrixForTraining) and isinstance(template, MLXMatrixForTraining):
        assert restored.weights.sharding == template.weights.sharding
    elif isinstance(restored, MLXMatrixForInference) and isinstance(template, MLXMatrixForInference):
        assert restored.packed_weights.sharding == template.packed_weights.sharding


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_mlx_from_packed_parameters_overrides_input_sharding(
    layout: Layout,
    implementation: CompressionImplementation,
) -> None:
    spec = MLXSpec(bits=4, group_size=2, layout=layout)
    original = spec.compress(_logical_weights(), implementation=CompressionImplementation.INFERENCE)
    template = spec.compress(
        dummy_array(_logical_weights().shape, _logical_weights().dtype),
        implementation=implementation,
    )
    saved_sharding = make_sharding((None, None))
    assert saved_sharding is not None
    assert isinstance(original, MLXMatrixForInference)

    restored = spec.from_packed_parameters(
        packed_weights=jax.device_put(original.packed_weights, saved_sharding),
        scales=jax.device_put(original.scales, saved_sharding),
        biases=jax.device_put(original.biases, saved_sharding),
        implementation=implementation,
    )

    _assert_close(result=restored.decompress(), reference=original.decompress())
    assert isinstance(restored, type(template))
    assert restored.scales.sharding == template.scales.sharding
    assert restored.biases.sharding == template.biases.sharding
    if isinstance(restored, MLXMatrixForTraining) and isinstance(template, MLXMatrixForTraining):
        assert restored.weights.sharding == template.weights.sharding
    elif isinstance(restored, MLXMatrixForInference) and isinstance(template, MLXMatrixForInference):
        assert restored.packed_weights.sharding == template.packed_weights.sharding


def _assert_mlx_replicated(matrix: MLXMatrix, fake_mesh: Mesh) -> None:
    replicated = make_sharding((None, None))
    assert replicated is not None

    _assert_named_sharding(matrix.scales.sharding, fake_mesh)
    assert matrix.scales.sharding == replicated
    assert matrix.biases.sharding == replicated
    if isinstance(matrix, MLXMatrixForTraining):
        assert matrix.weights.sharding == replicated
    else:
        assert isinstance(matrix, MLXMatrixForInference)
        assert matrix.packed_weights.sharding == replicated
    assert not matrix.is_sharded


@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_mlx_is_sharded_false_is_preserved_through_packed_conversions(
    fake_mesh: Mesh,
    implementation: CompressionImplementation,
) -> None:
    spec = MLXSpec(bits=4, group_size=2)
    original = spec.compress(_logical_weights(), implementation=CompressionImplementation.INFERENCE, is_sharded=False)
    assert isinstance(original, MLXMatrixForInference)

    restored = spec.from_packed_parameters(
        packed_weights=original.packed_weights,
        scales=original.scales,
        biases=original.biases,
        implementation=implementation,
        is_sharded=False,
    )
    switched = restored.switch_implementation(
        CompressionImplementation.INFERENCE
        if implementation == CompressionImplementation.TRAINING
        else CompressionImplementation.TRAINING,
    )
    full_precision = restored.to_full_precision()

    _assert_mlx_replicated(restored, fake_mesh)
    _assert_mlx_replicated(switched, fake_mesh)
    _assert_named_sharding(full_precision.weights.sharding, fake_mesh)
    assert full_precision.weights.sharding == make_sharding((None, None))
    assert not full_precision.is_sharded
    _assert_close(result=restored.decompress(), reference=original.decompress())
    _assert_close(result=full_precision.decompress(), reference=original.decompress())


def test_mlx_load_rejects_spec_mismatch() -> None:
    original = MLXSpec(bits=4, group_size=2, layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    template = MLXSpec(bits=8, group_size=2, layout=Layout.INPUT_OUTPUT).compress(
        dummy_array((4, 4), jnp.float32),
    )

    with pytest.raises(ValueError, match="spec mismatch"):
        template.load_exported(original.export())
