from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import Keychain, ShardingAxis
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixers.convolutions import SeparableCausalConv, SeparableCausalConvConfig
from lalamo.modules.token_mixers.short_conv import ShortConv, ShortConvConfig, ShortConvStateLayer
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec
from tests.common import assert_close

MODEL_DIM = 4
KERNEL_SIZE = 3


def _weights(shape: tuple[int, ...], *, offset: int = 0) -> jax.Array:
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 10) - 1


def _linear(weights: Array, output_dims: tuple[int, ...]) -> Linear:
    return Linear(
        config=LinearConfig(),
        weights=FullPrecisionSpec().compress(weights),
        biases=None,
        output_dims=output_dims,
    )


def _conv() -> SeparableCausalConv:
    return SeparableCausalConv(
        config=SeparableCausalConvConfig(has_biases=True),
        weights=_weights((MODEL_DIM, KERNEL_SIZE), offset=100),
        biases=jnp.array([-0.25, 0.0, 0.25, 0.5], dtype=jnp.float32),
    )


def _short_conv() -> ShortConv:
    return ShortConv(
        config=ShortConvConfig(
            in_projection_config=LinearConfig(),
            conv_config=SeparableCausalConvConfig(has_biases=True),
            out_projection_config=LinearConfig(),
            kernel_size=KERNEL_SIZE,
        ),
        in_projection=_linear(_weights((3 * MODEL_DIM, MODEL_DIM)), (MODEL_DIM, MODEL_DIM, MODEL_DIM)),
        conv=_conv(),
        out_projection=_linear(_weights((MODEL_DIM, MODEL_DIM), offset=200), (MODEL_DIM,)),
    )


def _linear_reference(linear: Linear, inputs: Array) -> tuple[Array, ...]:
    weights = linear.weights.decompress().astype(inputs.dtype)
    outputs = jnp.einsum("...i,oi->...o", inputs, weights)
    if linear.biases is not None:
        outputs = outputs + linear.biases.astype(outputs.dtype)
    return tuple(jnp.split(outputs, Linear.get_split_points(linear.output_dims), axis=-1))


def _conv_reference(module: SeparableCausalConv, inputs: Array, state: Array | None = None) -> Array:
    inputs = jnp.asarray(jax.device_get(inputs))
    weights = jnp.asarray(jax.device_get(module.weights)).astype(inputs.dtype)
    if state is None:
        state = jnp.zeros((module.kernel_size - 1, module.input_dim), dtype=inputs.dtype)
    else:
        state = jnp.asarray(jax.device_get(state))

    history = jnp.concatenate((state, inputs), axis=0)
    windows = jnp.stack([history[token : token + module.kernel_size] for token in range(inputs.shape[0])])
    result = jnp.einsum("tkc,ck->tc", windows, weights)
    if module.biases is not None:
        result = result + jnp.asarray(jax.device_get(module.biases)).astype(result.dtype)
    return result


def _reference(module: ShortConv, inputs: Array, state: ShortConvStateLayer | None = None) -> Array:
    inputs = jnp.asarray(jax.device_get(inputs))
    pre_conv_gate, post_conv_gate, conv_inputs = _linear_reference(module.in_projection, inputs)
    previous_state = state.conv_state if state is not None else None
    conv_outputs = _conv_reference(module.conv, conv_inputs * pre_conv_gate, previous_state)
    (outputs,) = _linear_reference(module.out_projection, conv_outputs * post_conv_gate)
    return outputs


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_sequence(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None, None)))


def _sharded_sequences(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA, None, None)))


def test_short_conv_matches_reference_and_keeps_unsharded_features(fake_mesh: Mesh) -> None:
    module = _short_conv()
    inputs = _sharded_sequence(jnp.arange(5 * MODEL_DIM, dtype=jnp.float32).reshape(5, MODEL_DIM) / 10)

    result = module(inputs, positional_embeddings=None, keychain=Keychain.init(0))

    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))
    assert result.state is None


def test_short_conv_updates_state_from_unpadded_suffix(fake_mesh: Mesh) -> None:
    module = _short_conv()
    state = ShortConvStateLayer(
        conv_state=_sharded_sequence(
            jnp.array([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]], dtype=jnp.float32),
        ),
    )
    inputs = _sharded_sequence(jnp.arange(4 * MODEL_DIM, dtype=jnp.float32).reshape(4, MODEL_DIM) / 10)

    result = module(
        inputs,
        positional_embeddings=None,
        state=state,
        return_updated_state=True,
        length_without_padding=2,
        keychain=Keychain.init(1),
    )

    assert result.state is not None
    _assert_close(result=result.outputs, reference=_reference(module, inputs, state))
    _assert_named_sharding(result.state.conv_state.sharding, fake_mesh)
    assert result.state.conv_state.sharding == make_sharding((None, None))


def test_short_conv_output_dtype_matches_input_dtype(fake_mesh: Mesh) -> None:
    module = _short_conv()
    inputs = _sharded_sequence(jnp.arange(5 * MODEL_DIM, dtype=jnp.bfloat16).reshape(5, MODEL_DIM) / 10)

    result = module(inputs, positional_embeddings=None, keychain=Keychain.init(7))

    assert result.outputs.dtype == inputs.dtype
    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)


@pytest.mark.usefixtures("fake_mesh")
def test_short_conv_rejects_positional_embeddings() -> None:
    module = _short_conv()
    inputs = jnp.arange(5 * MODEL_DIM, dtype=jnp.float32).reshape(5, MODEL_DIM) / 10
    positional_embeddings = PositionalEmbeddings(
        cosines=jnp.ones((5, MODEL_DIM), dtype=jnp.float32),
        sines=jnp.zeros((5, MODEL_DIM), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="not supported"):
        module(inputs, positional_embeddings=positional_embeddings, keychain=Keychain.init(2))


def test_short_conv_under_jit_matches_reference_and_keeps_unsharded_features(fake_mesh: Mesh) -> None:
    module = _short_conv()
    inputs = _sharded_sequence(jnp.arange(5 * MODEL_DIM, dtype=jnp.float32).reshape(5, MODEL_DIM) / 10)

    result = eqx.filter_jit(
        lambda module, values: module(values, positional_embeddings=None, keychain=Keychain.init(3)),
    )(module, inputs)

    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))


def test_short_conv_vmapped_over_inputs_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _short_conv()
    inputs = _sharded_sequences(jnp.arange(2 * 5 * MODEL_DIM, dtype=jnp.float32).reshape(2, 5, MODEL_DIM) / 10)

    result = call_vmapped(
        lambda values, *, keychain: module(values, positional_embeddings=None, keychain=keychain),
        inputs,
        keychain=Keychain.init(4),
        added_sharding_axis=ShardingAxis.DATA,
    )
    reference = jnp.stack([_reference(module, values) for values in jnp.asarray(jax.device_get(inputs))])

    _assert_close(result=result.outputs, reference=reference)
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_short_conv_export_load_roundtrips_with_replicated_conv_parameters(fake_mesh: Mesh) -> None:
    original = _short_conv()
    conv_weight_sharding = make_sharding((None, None))
    vector_sharding = make_sharding((None,))
    template = ShortConv(
        config=original.config,
        in_projection=Linear(
            config=LinearConfig(),
            weights=FullPrecisionSpec().compress(dummy_array(original.in_projection.weights.shape, jnp.float32)),
            biases=None,
            output_dims=original.in_projection.output_dims,
        ),
        conv=SeparableCausalConv(
            config=original.conv.config,
            weights=dummy_array(original.conv.weights.shape, jnp.float32, conv_weight_sharding),
            biases=dummy_array((MODEL_DIM,), jnp.float32, vector_sharding),
        ),
        out_projection=Linear(
            config=LinearConfig(),
            weights=FullPrecisionSpec().compress(dummy_array(original.out_projection.weights.shape, jnp.float32)),
            biases=None,
            output_dims=original.out_projection.output_dims,
        ),
    )
    inputs = _sharded_sequence(jnp.arange(5 * MODEL_DIM, dtype=jnp.float32).reshape(5, MODEL_DIM) / 10)

    restored = template.load_exported(original.export())
    result = restored(inputs, positional_embeddings=None, keychain=Keychain.init(5))

    assert isinstance(restored.in_projection.weights, FullPrecisionMatrix)
    assert isinstance(restored.out_projection.weights, FullPrecisionMatrix)
    assert isinstance(template.in_projection.weights, FullPrecisionMatrix)
    assert isinstance(template.out_projection.weights, FullPrecisionMatrix)
    assert restored.in_projection.weights.weights.sharding == template.in_projection.weights.weights.sharding
    assert restored.out_projection.weights.weights.sharding == template.out_projection.weights.weights.sharding
    assert restored.conv.weights.sharding == make_sharding((None, None))
    assert restored.conv.biases is not None
    assert template.conv.biases is not None
    assert restored.conv.biases.sharding == make_sharding((None,))
    _assert_named_sharding(restored.conv.weights.sharding, fake_mesh)
    _assert_named_sharding(restored.conv.biases.sharding, fake_mesh)
    _assert_close(
        result=result.outputs,
        reference=original(inputs, positional_embeddings=None, keychain=Keychain.init(6)).outputs,
    )
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))
