from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import ForwardPassMode, Keychain, ShardingAxis
from lalamo.modules.activations import Identity, SiLU
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.mlp import DenseMLP, DenseMLPConfig
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec
from tests.common import assert_close

MODEL_DIM = 4
HIDDEN_DIM = 4


def _array(shape: tuple[int, ...], *, offset: int = 0) -> jax.Array:
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 10) - 1


def _linear(weights: Array, biases: Array | None, output_dims: tuple[int, ...]) -> Linear:
    return Linear(
        config=LinearConfig(),
        weights=FullPrecisionSpec().compress(weights),
        biases=biases,
        output_dims=output_dims,
    )


def _dense_mlp(config: DenseMLPConfig | None = None) -> DenseMLP:
    if config is None:
        config = DenseMLPConfig(
            linear_config=LinearConfig(),
            activation=SiLU(),
            has_up_biases=True,
            has_down_biases=True,
            gate_clipping=(-0.5, 0.75),
            up_clipping=(-0.25, 0.5),
        )
    return DenseMLP(
        config=config,
        up_projection=_linear(
            _array((2 * HIDDEN_DIM, MODEL_DIM)),
            _array((2 * HIDDEN_DIM,), offset=100) if config.has_up_biases else None,
            (HIDDEN_DIM, HIDDEN_DIM),
        ),
        down_projection=_linear(
            _array((MODEL_DIM, HIDDEN_DIM), offset=200),
            _array((MODEL_DIM,), offset=300) if config.has_down_biases else None,
            (MODEL_DIM,),
        ),
    )


def _reference(module: DenseMLP, inputs: Array) -> Array:
    up_weights = module.up_projection.weights.decompress()
    down_weights = module.down_projection.weights.decompress()
    up_result = jnp.einsum("...i,oi->...o", inputs, up_weights)
    if module.up_projection.biases is not None:
        up_result = up_result + module.up_projection.biases
    up_projection, gate = jnp.split(up_result, (HIDDEN_DIM,), axis=-1)
    if module.config.gate_clipping is not None:
        gate = jnp.clip(gate, *module.config.gate_clipping)
    if module.config.up_clipping is not None:
        up_projection = jnp.clip(up_projection, *module.config.up_clipping)
    hidden = up_projection * module.config.activation(gate)
    result = jnp.einsum("...i,oi->...o", hidden, down_weights)
    if module.down_projection.biases is not None:
        result = result + module.down_projection.biases
    return result


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _sharded_vector(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.TENSOR,)))


def _sharded_vectors(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA, ShardingAxis.TENSOR)))


def _sharded_sequences(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA, None, ShardingAxis.TENSOR)))


def test_dense_mlp_call_unbatched_matches_reference_and_drops_tensor_sharding(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_vector(jnp.arange(MODEL_DIM, dtype=jnp.float32))

    result = module.call_unbatched(inputs, keychain=Keychain.init(0))

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_dense_mlp_call_unbatched_without_bias_or_clipping_matches_reference(fake_mesh: Mesh) -> None:
    module = _dense_mlp(
        DenseMLPConfig(
            linear_config=LinearConfig(),
            activation=Identity(),
            has_up_biases=False,
            has_down_biases=False,
            gate_clipping=None,
            up_clipping=None,
        ),
    )
    inputs = _sharded_vector(jnp.arange(MODEL_DIM, dtype=jnp.float32))

    result = module.call_unbatched(inputs, keychain=Keychain.init(1))

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_dense_mlp_under_jit_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_sequences(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)

    result = eqx.filter_jit(lambda module, values: module(values, keychain=Keychain.init(2)))(module, inputs)

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_dense_mlp_vmapped_over_inputs_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_vectors(jnp.arange(2 * MODEL_DIM, dtype=jnp.float32).reshape(2, MODEL_DIM) / 10)

    result = call_vmapped(
        module.call_unbatched,
        inputs,
        keychain=Keychain.init(3),
        added_sharding_axis=ShardingAxis.DATA,
    )

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))


def test_dense_mlp_full_call_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_sequences(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)

    result = module(
        inputs,
        forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        keychain=Keychain.init(4),
    )

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_dense_mlp_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _dense_mlp()
    weight_template = make_sharding((ShardingAxis.TENSOR, None))
    bias_template = make_sharding((ShardingAxis.DATA,))
    assert weight_template is not None
    assert bias_template is not None
    template = DenseMLP(
        config=original.config,
        up_projection=_linear(
            dummy_array(original.up_projection.weights.shape, jnp.float32, weight_template),
            dummy_array((2 * HIDDEN_DIM,), jnp.float32, bias_template),
            (HIDDEN_DIM, HIDDEN_DIM),
        ),
        down_projection=_linear(
            dummy_array(original.down_projection.weights.shape, jnp.float32, weight_template),
            dummy_array((MODEL_DIM,), jnp.float32, bias_template),
            (MODEL_DIM,),
        ),
    )
    inputs = _sharded_sequences(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)

    restored = template.load_exported(original.export())
    result = restored(inputs, keychain=Keychain.init(5))

    assert isinstance(restored.up_projection.weights, FullPrecisionMatrix)
    assert isinstance(restored.down_projection.weights, FullPrecisionMatrix)
    assert isinstance(template.up_projection.weights, FullPrecisionMatrix)
    assert isinstance(template.down_projection.weights, FullPrecisionMatrix)
    assert restored.up_projection.weights.weights.sharding == template.up_projection.weights.weights.sharding
    assert restored.down_projection.weights.weights.sharding == template.down_projection.weights.weights.sharding
    assert restored.up_projection.biases is not None
    assert template.up_projection.biases is not None
    assert restored.up_projection.biases.sharding == template.up_projection.biases.sharding
    _assert_close(result=result, reference=original(inputs, keychain=Keychain.init(6)))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None, None))
