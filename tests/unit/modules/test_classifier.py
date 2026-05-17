from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.activations import SiLU
from lalamo.modules.classifier import PredictionHead, PredictionHeadConfig
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig, UpcastMode
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config

FEATURE_DIM = 4
NUM_LABELS = 4


def _weights(*, offset: int = 0) -> jax.Array:
    shape = (NUM_LABELS, FEATURE_DIM)
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 10) - 1


def _biases(*, offset: int = 100) -> jax.Array:
    return (jnp.arange(offset, offset + NUM_LABELS, dtype=jnp.float32) / 20) - 2


def _linear(weights: Array, biases: Array | None) -> Linear:
    return Linear(
        config=LinearConfig(),
        sharding_config=make_test_sharding_config(),
        weights=FullPrecisionSpec().compress(weights, sharding_config=make_test_sharding_config()),
        biases=biases,
        output_dims=(NUM_LABELS,),
    )


def _normalization() -> Normalization:
    return Normalization(
        config=NormalizationConfig(
            epsilon=1e-5,
            scale_offset=0.25,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True,
            has_biases=True,
        ),
        sharding_config=make_test_sharding_config(),
        scales=jnp.array([1.0, 1.5, 2.0, 2.5], dtype=jnp.float32),
        biases=jnp.array([-0.5, -0.25, 0.25, 0.5], dtype=jnp.float32),
    )


def _prediction_head() -> PredictionHead:
    return PredictionHead(
        config=PredictionHeadConfig(
            dense_config=LinearConfig(),
            activation=SiLU(alpha=0.75),
            normalization_config=_normalization().config,
            readout_config=LinearConfig(),
            use_dense_bias=True,
        ),
        sharding_config=make_test_sharding_config(),
        dense=_linear(_weights(), _biases()),
        norm=_normalization(),
        readout=_linear(_weights(offset=200), _biases(offset=300)),
    )


def _normalization_reference(module: Normalization, inputs: Array) -> Array:
    upcasted = inputs.astype(jnp.float32)
    upcasted = upcasted - jnp.mean(upcasted)
    normalized = upcasted * jax.lax.rsqrt(jnp.mean(jnp.square(upcasted)) + module.config.epsilon)
    adjusted_scales = module.scales + 0.25
    result = normalized.astype(inputs.dtype) * adjusted_scales
    assert module.biases is not None
    return (result + module.biases).astype(inputs.dtype)


def _reference(module: PredictionHead, inputs: Array) -> Array:
    dense = _weights() @ inputs + _biases()
    activated = module.config.activation(dense)
    normalized = _normalization_reference(module.norm, activated)
    return _weights(offset=200) @ normalized + _biases(offset=300)


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_vector(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None,)))


def _sharded_vectors(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.BATCH, None)))


def test_prediction_head_call_unbatched_matches_reference_and_keeps_unsharded_features(fake_mesh: Mesh) -> None:
    module = _prediction_head()
    inputs = _sharded_vector(jnp.array([-1.0, -0.25, 0.5, 1.25], dtype=jnp.float32))

    result = module.call_unbatched(inputs, keychain=Keychain.init(0, sharding_config=make_test_sharding_config()))

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_prediction_head_under_jit_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _prediction_head()
    inputs = _sharded_vectors(jnp.arange(2 * FEATURE_DIM, dtype=jnp.float32).reshape(2, FEATURE_DIM) / 10)

    result = eqx.filter_jit(
        lambda module, values: module(
            values,
            keychain=Keychain.init(1, sharding_config=make_test_sharding_config()),
        ),
    )(module, inputs)
    reference = jax.vmap(lambda values: _reference(module, values))(inputs)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((LogicalAxis.BATCH, None))


def test_prediction_head_output_dtype_matches_input_dtype(fake_mesh: Mesh) -> None:
    module = _prediction_head()
    inputs = _sharded_vector(jnp.array([-1.0, -0.25, 0.5, 1.25], dtype=jnp.bfloat16))

    result = module.call_unbatched(inputs, keychain=Keychain.init(5, sharding_config=make_test_sharding_config()))

    assert result.dtype == inputs.dtype
    _assert_named_sharding(result.sharding, fake_mesh)


def test_prediction_head_vmapped_over_inputs_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _prediction_head()
    inputs = _sharded_vectors(jnp.arange(2 * FEATURE_DIM, dtype=jnp.float32).reshape(2, FEATURE_DIM) / 10)

    result = call_vmapped(
        module.call_unbatched,
        inputs,
        keychain=Keychain.init(2, sharding_config=make_test_sharding_config()),
        added_sharding_axis=make_test_sharding_config().resolve_axis(LogicalAxis.BATCH),
    )
    reference = jax.vmap(lambda values: _reference(module, values))(inputs)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((LogicalAxis.BATCH, None))


def test_prediction_head_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _prediction_head()
    weight_sharding = make_sharding((None, None))
    bias_sharding = make_sharding((None,))
    norm_sharding = make_sharding((None,))
    template = PredictionHead(
        config=original.config,
        sharding_config=make_test_sharding_config(),
        dense=Linear(
            config=LinearConfig(),
            sharding_config=make_test_sharding_config(),
            weights=FullPrecisionSpec().compress(
                dummy_array(_weights().shape, jnp.float32, weight_sharding),
                sharding_config=make_test_sharding_config(),
            ),
            biases=dummy_array((NUM_LABELS,), jnp.float32, bias_sharding),
            output_dims=(NUM_LABELS,),
        ),
        norm=Normalization(
            config=original.norm.config,
            sharding_config=make_test_sharding_config(),
            scales=dummy_array((FEATURE_DIM,), jnp.float32, norm_sharding),
            biases=dummy_array((FEATURE_DIM,), jnp.float32, norm_sharding),
        ),
        readout=Linear(
            config=LinearConfig(),
            sharding_config=make_test_sharding_config(),
            weights=FullPrecisionSpec().compress(
                dummy_array(_weights(offset=200).shape, jnp.float32, weight_sharding),
                sharding_config=make_test_sharding_config(),
            ),
            biases=dummy_array((NUM_LABELS,), jnp.float32, bias_sharding),
            output_dims=(NUM_LABELS,),
        ),
    )
    inputs = _sharded_vectors(jnp.arange(2 * FEATURE_DIM, dtype=jnp.float32).reshape(2, FEATURE_DIM) / 10)

    restored = template.load_exported(original.export())
    result = restored(inputs, keychain=Keychain.init(3, sharding_config=make_test_sharding_config()))
    reference = original(inputs, keychain=Keychain.init(4, sharding_config=make_test_sharding_config()))

    assert isinstance(restored.dense.weights, FullPrecisionMatrix)
    assert isinstance(restored.readout.weights, FullPrecisionMatrix)
    assert isinstance(template.dense.weights, FullPrecisionMatrix)
    assert isinstance(template.readout.weights, FullPrecisionMatrix)
    assert restored.dense.weights.weights.sharding == template.dense.weights.weights.sharding
    assert restored.readout.weights.weights.sharding == template.readout.weights.weights.sharding
    assert restored.norm.scales.sharding == template.norm.scales.sharding
    assert restored.norm.biases is not None
    assert template.norm.biases is not None
    assert restored.norm.biases.sharding == template.norm.biases.sharding
    _assert_named_sharding(restored.norm.scales.sharding, fake_mesh)
    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((LogicalAxis.BATCH, None))
