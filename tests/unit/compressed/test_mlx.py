from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, Sharding

from lalamo.compressed.mlx import MLXMatrixForInference, MLXMatrixForTraining, MLXSpec
from lalamo.compressed.utils.yaqa import yaqa_round_fixpoint
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import CompressionImplementation, Layout
from tests.common import assert_close_arrays, assert_named_sharding
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.compressed import test_common as compressed_common

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


def _put_on_sharding(matrix: MLXMatrixForInference, sharding: Sharding) -> MLXMatrixForInference:
    return MLXMatrixForInference(
        spec=matrix.spec,
        sharding_config=matrix.sharding_config,
        packed_weights=jax.device_put(matrix.packed_weights, sharding),
        scales=jax.device_put(matrix.scales, sharding),
        biases=jax.device_put(matrix.biases, sharding),
    )


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("bits", [4, 8])
def test_mlx_compress_and_decompress_match_manual_min_max_quantization(
    layout: Layout,
    bits: Literal[4, 8],
) -> None:
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

    training = spec.compress(
        weights, implementation=CompressionImplementation.TRAINING, sharding_config=make_test_sharding_config()
    )
    inference = spec.compress(
        weights, implementation=CompressionImplementation.INFERENCE, sharding_config=make_test_sharding_config()
    )

    assert isinstance(training, MLXMatrixForTraining)
    assert isinstance(inference, MLXMatrixForInference)
    assert inference.packed_weights.dtype == jnp.uint8
    assert inference.packed_weights.shape == compressed_common.expected_packed_shape(stored_weights.shape, bits)
    assert_close_arrays(result=training.scales, reference=expected_scales)
    assert_close_arrays(result=training.biases, reference=expected_biases)
    assert_close_arrays(result=inference.scales, reference=expected_scales)
    assert_close_arrays(result=inference.biases, reference=expected_biases)
    assert_close_arrays(result=training.decompress(), reference=expected_decompressed)
    assert_close_arrays(result=inference.decompress(), reference=expected_decompressed)


def test_mlx_compress_uses_yaqa_weights_when_preconditioned() -> None:
    weights = _preconditioned_weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = MLXSpec(bits=4, group_size=4)

    yaqa_weights = yaqa_round_fixpoint(
        weights,
        preconditioner,
        spec,
        sharding_config=make_test_sharding_config(),
    )
    preconditioned = spec.compress(weights, preconditioner=preconditioner, sharding_config=make_test_sharding_config())
    expected = spec.compress(yaqa_weights, sharding_config=make_test_sharding_config())

    assert_close_arrays(result=preconditioned.decompress(), reference=expected.decompress())


def test_mlx_compress_rejects_group_size_that_does_not_divide_stored_last_axis() -> None:
    weights = jnp.ones((4, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        MLXSpec(bits=4, group_size=2).compress(weights, sharding_config=make_test_sharding_config())


def test_mlx_export_load_roundtrips_and_preserves_template_sharding(
    fake_mesh: Mesh,
) -> None:
    weights = _logical_weights()
    spec = MLXSpec(bits=4, group_size=2, layout=Layout.INPUT_OUTPUT)
    saved_sharding = make_sharding((None, None))
    assert saved_sharding is not None
    original = spec.compress(
        weights, implementation=CompressionImplementation.INFERENCE, sharding_config=make_test_sharding_config()
    )
    assert isinstance(original, MLXMatrixForInference)
    original = _put_on_sharding(original, saved_sharding)
    template = spec.compress(
        dummy_array(weights.shape, weights.dtype, make_sharding((None, None))),
        implementation=CompressionImplementation.INFERENCE,
        sharding_config=make_test_sharding_config(),
    )

    restored = template.load_exported(original.export())

    assert restored.spec == spec
    assert isinstance(restored, type(template))
    assert_close_arrays(result=restored.decompress(), reference=original.decompress())
    assert_named_sharding(restored.scales.sharding, fake_mesh)
    assert restored.scales.sharding == template.scales.sharding
    assert restored.scales.sharding != saved_sharding
    assert restored.biases.sharding == template.biases.sharding
    assert isinstance(restored, MLXMatrixForInference)
    assert isinstance(template, MLXMatrixForInference)
    assert restored.packed_weights.sharding == template.packed_weights.sharding
