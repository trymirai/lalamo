from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.lax import DotAlgorithmPreset
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, MatmulConfig, ShapeDtypeMatrix
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config

INPUT_DIM = 4
OUTPUT_DIMS = (2, 2)
TOTAL_OUTPUT_DIM = sum(OUTPUT_DIMS)

MATMUL_PRECISIONS = [
    pytest.param(DotAlgorithmPreset.DEFAULT, id="default"),
    pytest.param(DotAlgorithmPreset.F32_F32_F32, id="f32"),
]


def _weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, TOTAL_OUTPUT_DIM, INPUT_DIM)
    return jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) / 10


def _biases(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, TOTAL_OUTPUT_DIM)
    return jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) / 5


def _linear(weights: Array, biases: Array | None) -> Linear:
    return Linear(
        config=LinearConfig(),
        sharding_config=make_test_sharding_config(),
        weights=FullPrecisionSpec().compress(weights, sharding_config=make_test_sharding_config()),
        biases=biases,
        output_dims=OUTPUT_DIMS,
    )


def _split_outputs(outputs: Array) -> tuple[Array, ...]:
    return tuple(jnp.split(outputs, Linear.get_split_points(OUTPUT_DIMS), axis=-1))


def _assert_close_outputs(outputs: tuple[Array, ...], reference: tuple[Array, ...]) -> None:
    assert len(outputs) == len(reference)
    for output, reference_output in zip(outputs, reference, strict=True):
        assert_close(
            result=jnp.asarray(jax.device_get(output)), reference=jnp.asarray(jax.device_get(reference_output))
        )


def _assert_outputs_sharded_like(outputs: tuple[Array, ...], sharding: Sharding | None, mesh: Mesh) -> None:
    assert sharding is not None
    for output in outputs:
        assert isinstance(output.sharding, NamedSharding)
        assert output.sharding == sharding
        assert output.sharding.mesh == mesh


def _sharded_input(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None,)))


def _sharded_batched_inputs(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.BATCH, None)))


def _sharded_expert_inputs(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.MIXTURE, None)))


@pytest.mark.parametrize("precision", MATMUL_PRECISIONS)
def test_linear_matches_reference_and_splits_outputs_with_biases(
    fake_mesh: Mesh,
    precision: DotAlgorithmPreset,
) -> None:
    linear = _linear(_weights(), _biases())
    inputs = _sharded_input(jnp.arange(INPUT_DIM, dtype=jnp.float32))
    forward_pass_config = MatmulConfig(precision=precision)

    outputs = linear(
        inputs,
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
        forward_pass_config=forward_pass_config,
    )

    reference = _split_outputs(_weights() @ inputs + _biases())
    _assert_close_outputs(outputs, reference)
    _assert_outputs_sharded_like(outputs, make_sharding((None,)), fake_mesh)


def test_linear_matches_reference_without_biases(fake_mesh: Mesh) -> None:
    linear = _linear(_weights(), None)
    inputs = _sharded_input(jnp.arange(INPUT_DIM, dtype=jnp.float32))

    outputs = linear(inputs, keychain=Keychain.init(1, sharding_config=make_test_sharding_config()))

    reference = _split_outputs(_weights() @ inputs)
    _assert_close_outputs(outputs, reference)
    _assert_outputs_sharded_like(outputs, make_sharding((None,)), fake_mesh)


def test_linear_output_dtype_matches_input_dtype_with_full_precision_parameters(fake_mesh: Mesh) -> None:
    linear = _linear(_weights(), _biases())
    inputs = jnp.arange(INPUT_DIM, dtype=jnp.bfloat16)

    outputs = linear(inputs, keychain=Keychain.init(8, sharding_config=make_test_sharding_config()))

    reference = _split_outputs(_weights().astype(inputs.dtype) @ inputs + _biases().astype(inputs.dtype))
    assert isinstance(linear.weights, FullPrecisionMatrix)
    assert isinstance(linear.weights.weights.sharding, NamedSharding)
    assert linear.weights.weights.sharding.mesh == fake_mesh
    for output in outputs:
        assert output.dtype == inputs.dtype
    _assert_close_outputs(outputs, reference)


@pytest.mark.parametrize("precision", MATMUL_PRECISIONS)
def test_linear_under_jit_matches_reference_and_keeps_unsharded_features(
    fake_mesh: Mesh,
    precision: DotAlgorithmPreset,
) -> None:
    linear = _linear(_weights(), _biases())
    inputs = _sharded_input(jnp.arange(INPUT_DIM, dtype=jnp.float32))
    forward_pass_config = MatmulConfig(precision=precision)

    @eqx.filter_jit
    def call(module: Linear, values: Array) -> tuple[Array, ...]:
        return module(
            values,
            keychain=Keychain.init(2, sharding_config=make_test_sharding_config()),
            forward_pass_config=forward_pass_config,
        )

    outputs = call(linear, inputs)

    reference = _split_outputs(_weights() @ inputs + _biases())
    _assert_close_outputs(outputs, reference)
    _assert_outputs_sharded_like(outputs, make_sharding((None,)), fake_mesh)


def test_linear_vmapped_over_inputs_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    linear = _linear(_weights(), _biases())
    inputs = _sharded_batched_inputs(jnp.arange(8, dtype=jnp.float32).reshape(2, INPUT_DIM))

    outputs = call_vmapped(
        linear,
        inputs,
        keychain=Keychain.init(3, sharding_config=make_test_sharding_config()),
        added_sharding_axis=make_test_sharding_config().resolve_axis(LogicalAxis.BATCH),
    )

    reference = _split_outputs(jnp.einsum("oi,bi->bo", _weights(), inputs) + _biases())
    _assert_close_outputs(outputs, reference)
    _assert_outputs_sharded_like(outputs, make_sharding((LogicalAxis.BATCH, None)), fake_mesh)


def test_linear_mixture_vmapped_over_experts_matches_reference_and_keeps_expert_sharding(fake_mesh: Mesh) -> None:
    linear = _linear(_weights(2), None)
    inputs = _sharded_expert_inputs(jnp.arange(8, dtype=jnp.float32).reshape(2, INPUT_DIM))

    with pytest.raises(ValueError, match="Use vmap"):
        linear(
            _sharded_input(jnp.arange(INPUT_DIM, dtype=jnp.float32)),
            keychain=Keychain.init(4, sharding_config=make_test_sharding_config()),
        )

    outputs = call_vmapped(
        lambda expert, values, *, keychain: expert(values, keychain=keychain),
        linear,
        inputs,
        keychain=Keychain.init(5, sharding_config=make_test_sharding_config()),
        in_axes=(0, 0),
        added_sharding_axis=make_test_sharding_config().resolve_axis(LogicalAxis.MIXTURE),
    )

    reference = _split_outputs(jnp.einsum("eoi,ei->eo", _weights(2), inputs))
    _assert_close_outputs(outputs, reference)
    _assert_outputs_sharded_like(outputs, make_sharding((LogicalAxis.MIXTURE, None)), fake_mesh)


def test_linear_config_init_mixture_keeps_biases_unsharded() -> None:
    linear = LinearConfig().init_mixture(
        EmptyInitializer(dtype=jnp.float32, sharding_config=make_test_sharding_config()),
        mixture_size=2,
        input_dim=INPUT_DIM,
        output_dims=OUTPUT_DIMS,
        has_biases=True,
    )

    assert linear.biases is not None
    assert linear.biases.sharding == make_sharding((None, None))


def test_linear_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _linear(_weights(), _biases())
    bias_template_sharding = make_sharding((None,))
    assert bias_template_sharding is not None
    template = Linear(
        config=LinearConfig(),
        sharding_config=make_test_sharding_config(),
        weights=EmptyInitializer(
            dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
        ).weight_matrix(TOTAL_OUTPUT_DIM, INPUT_DIM),
        biases=dummy_array((TOTAL_OUTPUT_DIM,), jnp.float32, bias_template_sharding),
        output_dims=OUTPUT_DIMS,
    )
    inputs = _sharded_input(jnp.arange(INPUT_DIM, dtype=jnp.float32))

    restored = template.load_exported(original.export())
    outputs = restored(inputs, keychain=Keychain.init(6, sharding_config=make_test_sharding_config()))

    assert isinstance(template.weights, ShapeDtypeMatrix)
    assert isinstance(restored.weights, FullPrecisionMatrix)
    assert restored.biases is not None
    assert template.biases is not None
    assert isinstance(restored.biases.sharding, NamedSharding)
    assert restored.biases.sharding == template.biases.sharding
    assert restored.biases.sharding.mesh == fake_mesh
    _assert_close_outputs(
        outputs, original(inputs, keychain=Keychain.init(7, sharding_config=make_test_sharding_config()))
    )
    _assert_outputs_sharded_like(outputs, make_sharding((None,)), fake_mesh)
