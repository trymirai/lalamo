from math import prod

import jax
import jax.numpy as jnp
import pytest
from einops import rearrange
from jax.sharding import Mesh, Sharding

from lalamo.compressed.microfloat import (
    MicrofloatMatrixForInference,
    MicrofloatMatrixForTraining,
    MicrofloatScaleMode,
    MicrofloatSpec,
)
from lalamo.compressed.utils.packing import pack_uint_to_uint8
from lalamo.compressed.utils.yaqa import yaqa_round_weights
from lalamo.model_import.loaders.utils import decode_mxfp4
from lalamo.module import Keychain, ShardingAxis
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, Layout
from tests.common import assert_close_arrays, assert_named_sharding

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 8)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 9) / 3


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


def _put_on_sharding(matrix: MicrofloatMatrixForInference, sharding: Sharding) -> MicrofloatMatrixForInference:
    return MicrofloatMatrixForInference(
        spec=matrix.spec,
        is_sharded=matrix.is_sharded,
        dtype_=matrix.dtype,
        packed_weights=jax.device_put(matrix.packed_weights, sharding),
        packed_scales=jax.device_put(matrix.packed_scales, sharding),
        global_scale=jax.device_put(matrix.global_scale, make_sharding((None,) * matrix.global_scale.ndim)),
    )


def test_microfloat_from_packed_parameters_decodes_fp4_values_in_gpt_oss_nibble_order() -> None:
    codes = jnp.arange(16, dtype=jnp.uint8).reshape(1, 16)
    packed_weights = pack_uint_to_uint8(codes, bits=4)
    scale_exponents = jnp.array([[127]], dtype=jnp.uint8)
    spec = MicrofloatSpec(group_size=16)
    expected = jnp.array(
        [
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        ],
        dtype=jnp.float32,
    )

    matrix = spec.from_packed_parameters(
        packed_weights=packed_weights,
        packed_scales=scale_exponents,
        global_scale=jnp.array(1, dtype=jnp.float32),
        dtype=jnp.float32,
        is_sharded=False,
    )

    assert isinstance(matrix, MicrofloatMatrixForInference)
    assert_close_arrays(result=matrix.decompress(), reference=expected)


def test_microfloat_from_packed_parameters_matches_existing_decode_mxfp4_helper() -> None:
    blocks = jnp.array([[[0x10, 0x32], [0x54, 0x76]]], dtype=jnp.uint8)
    scale_exponents = jnp.array([[127, 128]], dtype=jnp.uint8)
    spec = MicrofloatSpec(group_size=4)
    matrix = spec.from_packed_parameters(
        packed_weights=rearrange(blocks, "rows groups packed -> rows (groups packed)"),
        packed_scales=scale_exponents,
        global_scale=jnp.array(1, dtype=jnp.float32),
        dtype=jnp.float32,
        is_sharded=False,
    )
    expected = decode_mxfp4(blocks, scale_exponents, dtype=jnp.float32, flatten=False)
    expected = rearrange(expected, "rows groups group_size -> rows (groups group_size)")

    assert_close_arrays(result=matrix.decompress(), reference=expected)


def test_microfloat_compress_uses_e8m0_power_of_two_scale_exponents() -> None:
    weights = jnp.array([[0.0, 3.0, -6.0, 10.0]], dtype=jnp.float32)
    spec = MicrofloatSpec(group_size=4)

    matrix = spec.compress(weights, implementation=CompressionImplementation.TRAINING, is_sharded=False)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE, is_sharded=False)

    assert isinstance(matrix, MicrofloatMatrixForTraining)
    assert isinstance(inference, MicrofloatMatrixForInference)
    assert_close_arrays(result=matrix.master_scales, reference=jnp.array([[2.0]], dtype=jnp.float32))
    assert jnp.array_equal(inference.packed_scales, jnp.array([[128]], dtype=jnp.uint8))


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("scale_mode", list(MicrofloatScaleMode))
def test_microfloat_compress_training_and_inference_match(
    layout: Layout,
    scale_mode: MicrofloatScaleMode,
) -> None:
    weights = _logical_weights()
    spec = MicrofloatSpec(group_size=4, scale_mode=scale_mode, layout=layout)
    stored_weights = _stored_weights(layout, weights)

    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    inference = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)

    assert isinstance(training, MicrofloatMatrixForTraining)
    assert isinstance(inference, MicrofloatMatrixForInference)
    assert inference.packed_weights.dtype == jnp.uint8
    if scale_mode == MicrofloatScaleMode.MXFP4:
        assert inference.packed_scales.dtype == jnp.uint8
    else:
        assert inference.packed_scales.dtype == jnp.float8_e4m3fn
    assert inference.global_scale.shape == stored_weights.shape[:-2]
    assert inference.packed_weights.shape == (*stored_weights.shape[:-1], stored_weights.shape[-1] // 2)
    assert_close_arrays(result=training.decompress(), reference=inference.decompress())


@pytest.mark.parametrize("scale_mode", list(MicrofloatScaleMode))
def test_microfloat_training_dot_has_ste_gradients(scale_mode: MicrofloatScaleMode) -> None:
    weights = _logical_weights()
    vector = jnp.linspace(-1, 1, weights.shape[-1], dtype=weights.dtype)
    spec = MicrofloatSpec(group_size=4, scale_mode=scale_mode)
    training = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    assert isinstance(training, MicrofloatMatrixForTraining)

    def loss(
        master_weights: jax.Array,
        scales: jax.Array,
        global_scale: jax.Array,
    ) -> jax.Array:
        matrix = MicrofloatMatrixForTraining(
            spec=training.spec,
            is_sharded=training.is_sharded,
            master_weights=master_weights,
            master_scales=scales,
            global_scale=global_scale,
        )
        return jnp.sum(matrix.dot(vector, keychain=Keychain.init(17)))

    weight_gradients, scale_gradients, global_scale_gradients = jax.grad(loss, argnums=(0, 1, 2))(
        training.master_weights,
        training.master_scales,
        training.global_scale,
    )
    assert bool(jnp.any(weight_gradients != 0))
    assert bool(jnp.any(scale_gradients != 0))
    assert bool(jnp.any(global_scale_gradients != 0))


def test_microfloat_compress_uses_yaqa_weights_when_preconditioned() -> None:
    weights = _preconditioned_weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = MicrofloatSpec(group_size=4)

    yaqa_weights = yaqa_round_weights(weights, preconditioner, spec)
    preconditioned = spec.compress(weights, preconditioner=preconditioner)
    expected = spec.compress(yaqa_weights)

    assert_close_arrays(result=preconditioned.decompress(), reference=expected.decompress())


def test_microfloat_compress_rejects_group_size_that_does_not_divide_stored_last_axis() -> None:
    weights = jnp.ones((4, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        MicrofloatSpec(group_size=2).compress(weights)


def test_microfloat_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    weights = _logical_weights()
    spec = MicrofloatSpec(group_size=4, layout=Layout.INPUT_OUTPUT)
    saved_sharding = make_sharding((ShardingAxis.TENSOR, None))
    assert saved_sharding is not None
    original = spec.compress(weights, implementation=CompressionImplementation.INFERENCE)
    assert isinstance(original, MicrofloatMatrixForInference)
    original = _put_on_sharding(original, saved_sharding)
    template = spec.compress(
        dummy_array(weights.shape, weights.dtype),
        implementation=CompressionImplementation.INFERENCE,
    )

    restored = template.load_exported(original.export())

    assert restored.spec == spec
    assert isinstance(restored, type(template))
    assert isinstance(restored, MicrofloatMatrixForInference)
    assert isinstance(template, MicrofloatMatrixForInference)
    assert_close_arrays(result=restored.decompress(), reference=original.decompress())
    assert_named_sharding(restored.packed_scales.sharding, fake_mesh)
    assert restored.packed_scales.sharding == template.packed_scales.sharding
    assert restored.packed_scales.sharding != saved_sharding
    assert restored.packed_weights.sharding == template.packed_weights.sharding
