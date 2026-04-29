from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import ShardingAxis
from lalamo.modules.token_mixers.convolutions import SeparableCausalConv, SeparableCausalConvConfig
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from tests.common import assert_close

CHANNELS = 4
KERNEL_SIZE = 3


def _weights() -> jax.Array:
    shape = (CHANNELS, KERNEL_SIZE)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) / 10) - 0.5


def _biases() -> jax.Array:
    return jnp.array([-0.25, 0.0, 0.25, 0.5], dtype=jnp.float32)


def _conv(has_biases: bool = True) -> SeparableCausalConv:
    return SeparableCausalConv(
        config=SeparableCausalConvConfig(has_biases=has_biases),
        weights=_weights(),
        biases=_biases() if has_biases else None,
    )


def _reference(module: SeparableCausalConv, inputs: Array, state: Array | None = None) -> Array:
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


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_sequence(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None, ShardingAxis.TENSOR)))


def _sharded_sequences(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA, None, ShardingAxis.TENSOR)))


def test_separable_causal_conv_matches_reference_and_drops_tensor_sharding(fake_mesh: Mesh) -> None:
    module = _conv()
    inputs = _sharded_sequence(jnp.arange(5 * CHANNELS, dtype=jnp.float32).reshape(5, CHANNELS) / 10)

    result = module(inputs)

    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))
    assert result.state is None


def test_separable_causal_conv_without_biases_matches_reference(fake_mesh: Mesh) -> None:
    module = _conv(has_biases=False)
    inputs = _sharded_sequence(jnp.arange(5 * CHANNELS, dtype=jnp.float32).reshape(5, CHANNELS) / 10)

    result = module(inputs)

    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))


def test_separable_causal_conv_output_dtype_matches_input_dtype(fake_mesh: Mesh) -> None:
    module = _conv()
    inputs = _sharded_sequence(jnp.arange(5 * CHANNELS, dtype=jnp.bfloat16).reshape(5, CHANNELS) / 10)

    result = module(inputs)

    assert result.outputs.dtype == inputs.dtype
    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)


def test_separable_causal_conv_updates_state_from_unpadded_suffix(fake_mesh: Mesh) -> None:
    module = _conv()
    state = _sharded_sequence(jnp.array([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]], dtype=jnp.float32))
    inputs = _sharded_sequence(jnp.arange(4 * CHANNELS, dtype=jnp.float32).reshape(4, CHANNELS) / 10)

    result = module(inputs, length_without_padding=2, state=state, return_updated_state=True)

    assert result.state is not None
    _assert_close(result=result.outputs, reference=_reference(module, inputs, state))
    _assert_close(result=result.state, reference=jnp.asarray(jax.device_get(inputs))[:2])
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    _assert_named_sharding(result.state.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))
    assert result.state.sharding == make_sharding((None, None))


def test_separable_causal_conv_step_matches_last_full_call_output() -> None:
    module = _conv()
    state = jnp.array([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]], dtype=jnp.float32)
    token = jnp.array([0.0, 0.25, 0.5, 0.75], dtype=jnp.float32)

    step_output, next_state = module.step(token, state)
    full_output = module(token[None, :], state=state, return_updated_state=True)

    _assert_close(result=step_output, reference=full_output.outputs[0])
    assert full_output.state is not None
    _assert_close(result=next_state, reference=full_output.state)


def test_separable_causal_conv_step_output_dtype_matches_token_dtype() -> None:
    module = _conv()
    state = jnp.array([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]], dtype=jnp.bfloat16)
    token = jnp.array([0.0, 0.25, 0.5, 0.75], dtype=jnp.bfloat16)

    step_output, _ = module.step(token, state)

    assert step_output.dtype == token.dtype


def test_separable_causal_conv_under_jit_matches_reference_and_drops_tensor_sharding(fake_mesh: Mesh) -> None:
    module = _conv()
    inputs = _sharded_sequence(jnp.arange(5 * CHANNELS, dtype=jnp.float32).reshape(5, CHANNELS) / 10)

    result = eqx.filter_jit(lambda module, values: module(values))(module, inputs)

    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))


def test_separable_causal_conv_vmapped_over_inputs_matches_reference_and_keeps_data_sharding(
    fake_mesh: Mesh,
) -> None:
    module = _conv()
    inputs = _sharded_sequences(jnp.arange(2 * 5 * CHANNELS, dtype=jnp.float32).reshape(2, 5, CHANNELS) / 10)

    result = call_vmapped(module, inputs, added_sharding_axis=ShardingAxis.DATA)
    reference = jnp.stack([_reference(module, values) for values in jnp.asarray(jax.device_get(inputs))])

    _assert_close(result=result.outputs, reference=reference)
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_separable_causal_conv_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _conv()
    weight_sharding = make_sharding((ShardingAxis.DATA, None))
    bias_sharding = make_sharding((ShardingAxis.DATA,))
    template = SeparableCausalConv(
        config=original.config,
        weights=dummy_array(original.weights.shape, original.weights.dtype, weight_sharding),
        biases=dummy_array(original.biases.shape, original.biases.dtype, bias_sharding)
        if original.biases is not None
        else None,
    )
    inputs = _sharded_sequence(jnp.arange(5 * CHANNELS, dtype=jnp.float32).reshape(5, CHANNELS) / 10)

    restored = template.load_exported(original.export())
    result = restored(inputs)

    assert restored.weights.sharding == template.weights.sharding
    assert restored.biases is not None
    assert template.biases is not None
    assert restored.biases.sharding == template.biases.sharding
    _assert_named_sharding(restored.weights.sharding, fake_mesh)
    _assert_named_sharding(restored.biases.sharding, fake_mesh)
    _assert_close(result=result.outputs, reference=original(inputs).outputs)
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))
