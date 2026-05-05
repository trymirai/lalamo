from functools import partial
from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding

from lalamo.compressed.mlx import (
    MLXMatrix,
    MLXMatrixForInference,
    MLXMatrixForTraining,
    MLXSpec,
    _mlx_quantize,
    _mlx_quantized_dot_output_input,
)
from lalamo.compressed.rounding import deterministic_round_to_unsigned_grid
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, Layout, Preconditioner, WeightMatrixSpec
from tests.common import assert_close

type Bits = Literal[4, 8]

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 4)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 3) / 5


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
            weights=jax.device_put(matrix.weights, sharding),
            scales=jax.device_put(matrix.scales, sharding),
            biases=jax.device_put(matrix.biases, sharding),
        )
    assert isinstance(matrix, MLXMatrixForInference)
    return MLXMatrixForInference(
        spec=matrix.spec,
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


def test_mlx_training_grouped_dot_matches_materialized_forward_and_gradients() -> None:
    bits = 4
    group_size = 2
    weights = _stored_weights(Layout.OUTPUT_INPUT, _logical_weights()).astype(jnp.bfloat16)
    scales, biases = _manual_mlx_affine_parameters(weights, bits=bits, group_size=group_size)
    vector = jnp.array([0.4, -0.2, 0.7, -0.5], dtype=jnp.bfloat16)
    cotangent = jnp.array([1.0, -0.5, 0.25, -0.75], dtype=jnp.float32)
    round_fn = partial(deterministic_round_to_unsigned_grid, bits=bits)

    def grouped_loss(
        weights: jax.Array,
        scales: jax.Array,
        biases: jax.Array,
        vector: jax.Array,
    ) -> jax.Array:
        result = _mlx_quantized_dot_output_input(weights, scales, biases, vector, group_size, round_fn)
        return jnp.sum(result.astype(jnp.float32) * cotangent)

    def materialized_loss(
        weights: jax.Array,
        scales: jax.Array,
        biases: jax.Array,
        vector: jax.Array,
    ) -> jax.Array:
        quantized = _mlx_quantize(weights, scales, biases, group_size=group_size, round_fn=round_fn)
        result = quantized @ vector
        return jnp.sum(result.astype(jnp.float32) * cotangent)

    grouped_value, grouped_grads = jax.value_and_grad(grouped_loss, argnums=(0, 1, 2, 3))(
        weights,
        scales,
        biases,
        vector,
    )
    materialized_value, materialized_grads = jax.value_and_grad(materialized_loss, argnums=(0, 1, 2, 3))(
        weights,
        scales,
        biases,
        vector,
    )

    assert_close(result=grouped_value[None], reference=materialized_value[None])
    for grouped_grad, materialized_grad in zip(grouped_grads, materialized_grads, strict=True):
        assert_close(result=grouped_grad, reference=materialized_grad)


@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_mlx_output_input_dot_vmapped_over_bfloat16_inputs_matches_decompressed(
    implementation: CompressionImplementation,
) -> None:
    matrix = MLXSpec(bits=4, group_size=2, layout=Layout.OUTPUT_INPUT).compress(
        _logical_weights(),
        implementation=implementation,
    )
    vectors = jnp.asarray(
        [
            [0.4, -0.2, 0.7, -0.5],
            [-0.1, 0.3, -0.6, 0.8],
        ],
        dtype=jnp.bfloat16,
    )
    reference_weights = matrix.astype(vectors.dtype).decompress()

    result = jax.vmap(lambda vector: matrix.dot(vector, keychain=Keychain.init(0)))(vectors)
    reference = jax.vmap(lambda vector: reference_weights @ vector)(vectors)

    assert_close(result=result, reference=reference)


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


def test_mlx_compress_rejects_preconditioning() -> None:
    preconditioner = Preconditioner.init(input_block=jnp.eye(4, dtype=jnp.float32))

    with pytest.raises(ValueError, match="Preconditioned rounding is not implemented"):
        MLXSpec(bits=4, group_size=2).compress(_logical_weights(), preconditioner=preconditioner)


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


def test_mlx_load_rejects_spec_mismatch() -> None:
    original = MLXSpec(bits=4, group_size=2, layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    template = MLXSpec(bits=8, group_size=2, layout=Layout.INPUT_OUTPUT).compress(
        dummy_array((4, 4), jnp.float32),
    )

    with pytest.raises(ValueError, match="spec mismatch"):
        template.load_exported(original.export())
