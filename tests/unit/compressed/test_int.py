from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, Sharding

from lalamo.compressed.int import IntMatrixForInference, IntMatrixForTraining, IntSpec
from lalamo.compressed.utils.yaqa import yaqa_round_fixpoint
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import is_sharded
from lalamo.weight_matrix import CompressionImplementation, Layout
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.compressed import test_common as compressed_common

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


def _manual_int_affine_parameters(
    stored_weights: jax.Array,
    *,
    bits: int,
    group_size: int,
    is_symmetric: bool,
) -> tuple[jax.Array, jax.Array | None]:
    *leading_dims, rows, cols = stored_weights.shape
    grouped = stored_weights.reshape(*leading_dims, rows, cols // group_size, group_size)
    min_values = jnp.min(grouped, axis=-1)
    max_values = jnp.max(grouped, axis=-1)
    if is_symmetric:
        scales = jnp.nan_to_num(
            jnp.maximum(jnp.abs(min_values), jnp.abs(max_values)) / ((2 ** (bits - 1)) - 1),
            nan=jnp.finfo(stored_weights.dtype).eps,
            posinf=jnp.finfo(stored_weights.dtype).max,
            neginf=jnp.finfo(stored_weights.dtype).eps,
        )
        zero_points = None
    else:
        scales = jnp.maximum((max_values - min_values) / ((2**bits) - 1), jnp.finfo(stored_weights.dtype).eps)
        zero_points = jnp.nan_to_num(-min_values, nan=0, posinf=jnp.finfo(stored_weights.dtype).max, neginf=0)
    return scales, zero_points


def _round_unsigned(values: jax.Array, *, bits: int) -> jax.Array:
    return jnp.round(jnp.clip(values, 0, (2**bits) - 1))


def _manual_int_dequantize(
    stored_weights: jax.Array,
    scales: jax.Array,
    zero_points: jax.Array | None,
    *,
    bits: int,
    group_size: int,
) -> jax.Array:
    expanded_scales = jnp.repeat(scales, group_size, axis=-1)
    if zero_points is None:
        int_weights = _round_unsigned(stored_weights / expanded_scales + 2 ** (bits - 1), bits=bits)
        return (int_weights - 2 ** (bits - 1)) * expanded_scales

    int_scale_zero_points = _round_unsigned(zero_points / scales, bits=bits)
    rounded_zero_points = int_scale_zero_points * scales
    expanded_rounded_zero_points = jnp.repeat(rounded_zero_points, group_size, axis=-1)
    expanded_int_scale_zero_points = jnp.repeat(int_scale_zero_points, group_size, axis=-1)
    int_weights = _round_unsigned((stored_weights + expanded_rounded_zero_points) / expanded_scales, bits=bits)
    return (int_weights - expanded_int_scale_zero_points) * expanded_scales


def _put_on_sharding(matrix: IntMatrixForInference, sharding: Sharding) -> IntMatrixForInference:
    packed_zero_points = None
    if matrix.packed_zero_points is not None:
        packed_zero_points = jax.device_put(matrix.packed_zero_points, sharding)
    return IntMatrixForInference(
        spec=matrix.spec,
        sharding_config=matrix.sharding_config,
        is_sharded=matrix.is_sharded,
        packed_weights=jax.device_put(matrix.packed_weights, sharding),
        scales=jax.device_put(matrix.scales, sharding),
        packed_zero_points=packed_zero_points,
    )


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("is_symmetric", [False, True])
def test_int_compress_and_decompress_match_manual_quantization(
    layout: Layout,
    bits: Literal[4, 8],
    is_symmetric: bool,
) -> None:
    weights = _logical_weights()
    spec = IntSpec(bits=bits, group_size=2, is_symmetric=is_symmetric, layout=layout)
    stored_weights = _stored_weights(layout, weights)
    expected_scales, expected_zero_points = _manual_int_affine_parameters(
        stored_weights,
        bits=bits,
        group_size=2,
        is_symmetric=is_symmetric,
    )
    expected_decompressed = _manual_int_dequantize(
        stored_weights,
        expected_scales,
        expected_zero_points,
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

    assert isinstance(training, IntMatrixForTraining)
    assert isinstance(inference, IntMatrixForInference)
    assert inference.packed_weights.dtype == jnp.uint8
    assert inference.packed_weights.shape == compressed_common.expected_packed_shape(stored_weights.shape, bits)
    compressed_common.assert_close_arrays(result=training.scales, reference=expected_scales)
    compressed_common.assert_close_arrays(result=inference.scales, reference=expected_scales)
    if is_symmetric:
        assert training.master_zero_points is None
        assert inference.packed_zero_points is None
    else:
        assert expected_zero_points is not None
        assert training.master_zero_points is not None
        assert inference.packed_zero_points is not None
        assert inference.packed_zero_points.dtype == jnp.uint8
        assert inference.packed_zero_points.shape == compressed_common.expected_packed_shape(
            expected_scales.shape,
            bits,
        )
        compressed_common.assert_close_arrays(result=training.master_zero_points, reference=expected_zero_points)
    compressed_common.assert_close_arrays(result=training.decompress(), reference=expected_decompressed)
    compressed_common.assert_close_arrays(result=inference.decompress(), reference=expected_decompressed)


def test_int_compress_uses_yaqa_weights_when_preconditioned() -> None:
    weights = _preconditioned_weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = IntSpec(bits=4, group_size=2)

    yaqa_weights = yaqa_round_fixpoint(
        weights,
        preconditioner,
        spec,
        sharding_config=make_test_sharding_config(),
    )
    preconditioned = spec.compress(weights, preconditioner=preconditioner, sharding_config=make_test_sharding_config())
    expected = spec.compress(yaqa_weights, sharding_config=make_test_sharding_config())

    compressed_common.assert_close_arrays(result=preconditioned.decompress(), reference=expected.decompress())


def test_int_compress_rejects_group_size_that_does_not_divide_stored_last_axis() -> None:
    weights = jnp.ones((4, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        IntSpec(bits=4, group_size=2).compress(weights, sharding_config=make_test_sharding_config())


def test_int_export_load_roundtrips_and_preserves_template_sharding(
    fake_mesh: Mesh,
) -> None:
    weights = _logical_weights()
    spec = IntSpec(bits=4, group_size=2, layout=Layout.INPUT_OUTPUT)
    saved_sharding = make_sharding((None, None))
    assert saved_sharding is not None
    original = spec.compress(
        weights, implementation=CompressionImplementation.INFERENCE, sharding_config=make_test_sharding_config()
    )
    assert isinstance(original, IntMatrixForInference)
    original = _put_on_sharding(original, saved_sharding)
    template = spec.compress(
        dummy_array(weights.shape, weights.dtype, make_sharding((None, None))),
        implementation=CompressionImplementation.INFERENCE,
        sharding_config=make_test_sharding_config(),
    )

    restored = template.load_exported(original.export())

    assert restored.spec == spec
    assert isinstance(restored, type(template))
    compressed_common.assert_close_arrays(result=restored.decompress(), reference=original.decompress())
    compressed_common.assert_named_sharding(restored.scales.sharding, fake_mesh)
    assert restored.scales.sharding == template.scales.sharding
    assert restored.scales.sharding != saved_sharding
    assert isinstance(restored, IntMatrixForInference)
    assert isinstance(template, IntMatrixForInference)
    assert restored.packed_weights.sharding == template.packed_weights.sharding
    assert restored.packed_zero_points is not None
    assert template.packed_zero_points is not None
    assert not is_sharded(restored.packed_zero_points.sharding)


def test_int_symmetric_from_packed_parameters_rejects_zero_points() -> None:
    original = IntSpec(bits=4, group_size=2, layout=Layout.INPUT_OUTPUT).compress(
        _logical_weights(),
        implementation=CompressionImplementation.INFERENCE,
        sharding_config=make_test_sharding_config(),
    )
    spec = IntSpec(bits=4, group_size=2, is_symmetric=True, layout=Layout.INPUT_OUTPUT)
    assert isinstance(original, IntMatrixForInference)
    assert original.packed_zero_points is not None

    with pytest.raises(ValueError, match="must not include packed zero points"):
        spec.from_packed_parameters(
            packed_weights=original.packed_weights,
            scales=original.scales,
            packed_zero_points=original.packed_zero_points,
            sharding_config=make_test_sharding_config(),
        )
