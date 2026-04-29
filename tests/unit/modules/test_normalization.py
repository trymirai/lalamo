import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import ShardingAxis
from lalamo.modules.normalization import Normalization, NormalizationConfig, UpcastMode
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from tests.common import assert_close


def _config(
    *,
    subtract_mean: bool = True,
    scale_offset: float | None = 0.25,
    upcast_mode: UpcastMode = UpcastMode.ONLY_NORMALIZATION,
    has_biases: bool = True,
) -> NormalizationConfig:
    return NormalizationConfig(
        epsilon=1e-5,
        scale_offset=scale_offset,
        upcast_mode=upcast_mode,
        subtract_mean=subtract_mean,
        has_biases=has_biases,
    )


def _normalization(config: NormalizationConfig | None = None) -> Normalization:
    if config is None:
        config = _config()
    scales = jnp.array([1.0, 1.5, 2.0, 2.5], dtype=jnp.float32)
    biases = jnp.array([-0.25, 0.0, 0.25, 0.5], dtype=jnp.float32) if config.has_biases else None
    return Normalization(config=config, scales=scales, biases=biases)


def _reference(module: Normalization, inputs: Array) -> Array:
    config = module.config
    upcasted_inputs = inputs.astype(jnp.float32)
    if config.subtract_mean:
        upcasted_inputs = upcasted_inputs - jnp.mean(upcasted_inputs)

    normalized = upcasted_inputs * jax.lax.rsqrt(jnp.mean(jnp.square(upcasted_inputs)) + config.epsilon)
    if config.upcast_mode == UpcastMode.ONLY_NORMALIZATION:
        normalized = normalized.astype(inputs.dtype)

    if config.upcast_mode == UpcastMode.FULL_LAYER:
        adjusted_scales = module.scales.astype(jnp.float32)
    else:
        adjusted_scales = module.scales

    if config.scale_offset is not None:
        adjusted_scales = adjusted_scales + config.scale_offset

    result = normalized * adjusted_scales
    if module.biases is not None:
        result = result + module.biases
    return result.astype(inputs.dtype)


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_input(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.TENSOR,)))


def _sharded_batched_inputs(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA, ShardingAxis.TENSOR)))


def test_normalization_matches_reference_and_drops_tensor_sharding(fake_mesh: Mesh) -> None:
    module = _normalization()
    inputs = _sharded_input(jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32))

    result = module(inputs)

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_normalization_without_mean_offset_or_biases_matches_reference(fake_mesh: Mesh) -> None:
    module = _normalization(_config(subtract_mean=False, scale_offset=None, has_biases=False))
    inputs = _sharded_input(jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32))

    result = module(inputs)

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_normalization_full_layer_upcast_matches_reference_and_returns_input_dtype(fake_mesh: Mesh) -> None:
    module = _normalization(_config(upcast_mode=UpcastMode.FULL_LAYER))
    inputs = _sharded_input(jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float16))

    result = module(inputs)

    assert result.dtype == inputs.dtype
    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_normalization_under_jit_matches_reference_and_drops_tensor_sharding(fake_mesh: Mesh) -> None:
    module = _normalization()
    inputs = _sharded_input(jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32))

    result = eqx.filter_jit(lambda module, values: module(values))(module, inputs)

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_normalization_vmapped_over_inputs_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _normalization()
    inputs = _sharded_batched_inputs(
        jnp.array(
            [
                [1.0, -2.0, 3.0, -4.0],
                [2.0, -1.0, 4.0, -3.0],
            ],
            dtype=jnp.float32,
        ),
    )

    result = call_vmapped(module, inputs, added_sharding_axis=ShardingAxis.DATA)
    reference = jax.vmap(lambda values: _reference(module, values))(inputs)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))


def test_normalization_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _normalization()
    parameter_sharding = make_sharding((ShardingAxis.DATA,))
    template = Normalization(
        config=original.config,
        scales=dummy_array(original.scales.shape, original.scales.dtype, parameter_sharding),
        biases=dummy_array(original.biases.shape, original.biases.dtype, parameter_sharding)
        if original.biases is not None
        else None,
    )
    inputs = _sharded_input(jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32))

    restored = template.load_exported(original.export())
    result = restored(inputs)

    assert restored.scales.sharding == template.scales.sharding
    assert isinstance(restored.scales.sharding, NamedSharding)
    assert restored.scales.sharding.mesh == fake_mesh
    assert restored.biases is not None
    assert template.biases is not None
    assert restored.biases.sharding == template.biases.sharding
    _assert_close(result=result, reference=original(inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))
