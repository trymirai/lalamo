import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from lalamo.modules.normalization import (
    Normalization,
    NormalizationConfig,
    NormalizationForwardPassConfig,
    NormalizationImplementation,
    UpcastMode,
)
from tests.common import assert_close


def _normalization(*, upcast_mode: UpcastMode = UpcastMode.ONLY_NORMALIZATION) -> Normalization:
    return Normalization(
        config=NormalizationConfig(
            epsilon=1e-5,
            scale_offset=0.25,
            upcast_mode=upcast_mode,
            subtract_mean=True,
            has_biases=True,
        ),
        scales=jnp.array([1.0, 1.5, 2.0, 2.5], dtype=jnp.float32),
        biases=jnp.array([-0.25, 0.0, 0.25, 0.5], dtype=jnp.float32),
    )


def _reference(module: Normalization, inputs: Array) -> Array:
    centered = inputs.astype(jnp.float32) - jnp.mean(inputs.astype(jnp.float32))
    normalized = centered * jax.lax.rsqrt(jnp.mean(jnp.square(centered)) + module.config.epsilon)
    if module.config.upcast_mode == UpcastMode.ONLY_NORMALIZATION:
        normalized = normalized.astype(inputs.dtype)
        scales = module.scales.astype(inputs.dtype)
    else:
        scales = module.scales.astype(jnp.float32)
    assert module.biases is not None
    result = normalized * (scales + 0.25) + module.biases.astype(normalized.dtype)
    return result.astype(inputs.dtype)


def _call(module: Normalization, inputs: Array, implementation: NormalizationImplementation) -> Array:
    return module(
        inputs,
        forward_pass_config=NormalizationForwardPassConfig(implementation=implementation),
    )


@pytest.mark.parametrize("upcast_mode", (UpcastMode.ONLY_NORMALIZATION, UpcastMode.FULL_LAYER))
def test_normalization_matches_reference(upcast_mode: UpcastMode) -> None:
    module = _normalization(upcast_mode=upcast_mode)
    inputs = jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)

    assert_close(result=module(inputs), reference=_reference(module, inputs))


def test_tokamax_normalization_is_not_less_precise_than_standard() -> None:
    module = _normalization()
    inputs = jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.bfloat16)
    reference = module(inputs.astype(jnp.float32)).astype(inputs.dtype)

    standard_error = jnp.max(jnp.abs(_call(module, inputs, NormalizationImplementation.STANDARD) - reference))
    tokamax_error = jnp.max(jnp.abs(_call(module, inputs, NormalizationImplementation.TOKAMAX) - reference))

    assert tokamax_error <= standard_error
