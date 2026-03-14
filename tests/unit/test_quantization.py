from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp

from lalamo.modules.linear import GroupQuantizedLinearConfig, MLXQuantizedLinearConfig
from lalamo.quantization import QuantizationMode, quantize_weights, stochastic_quantize_weights


def test_quantize_weights_primal_matches_round_clip_reference() -> None:
    x = jnp.array([-129.4, -128.0, -0.6, -0.5, 0.49, 0.5, 14.5, 15.0, 15.1, 255.4], dtype=jnp.float32)

    for mode in QuantizationMode:
        range_min, range_max = mode.range
        expected = jnp.clip(jnp.round(x), range_min, range_max)
        result = quantize_weights(x, mode)

        assert result.dtype == expected.dtype
        assert jnp.array_equal(result, expected)


def test_quantize_weights_uses_clipped_ste_gradient() -> None:
    x = jnp.array([-1.0, 0.0, 0.49, 7.5, 15.0, 15.1, 16.0], dtype=jnp.float32)

    grad = jax.grad(lambda weights: jnp.sum(quantize_weights(weights, QuantizationMode.UINT4)))(x)

    expected = jnp.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    assert jnp.array_equal(grad, expected)


def test_stochastic_quantize_weights_is_deterministic_for_fixed_key() -> None:
    x = jnp.array([-1.0, 0.2, 0.75, 7.5, 14.25, 15.9], dtype=jnp.float32)
    key = jax.random.key(0)

    first = stochastic_quantize_weights(x, QuantizationMode.UINT4, key)
    second = stochastic_quantize_weights(x, QuantizationMode.UINT4, key)

    assert jnp.array_equal(first, second)


def test_stochastic_quantize_weights_matches_input_in_expectation() -> None:
    x = jnp.array([0.2, 0.75, 7.5, 14.25], dtype=jnp.float32)
    keys = jax.random.split(jax.random.key(1), 4096)
    samples = jax.vmap(lambda key: stochastic_quantize_weights(x, QuantizationMode.UINT4, key))(keys)
    means = samples.mean(axis=0)

    assert jnp.allclose(means, x, atol=0.05)


def test_quantized_linear_weights_have_nonzero_gradients() -> None:
    input_vector = jnp.array([0.25, -0.5, 1.0, 0.75], dtype=jnp.float32)
    configs = [
        GroupQuantizedLinearConfig(
            group_size=2,
            weight_quantization_mode=QuantizationMode.UINT4,
            activation_quantization_mode=None,
            activation_precision=jnp.float32,
        ),
        MLXQuantizedLinearConfig(
            group_size=2,
            weight_quantization_mode=QuantizationMode.UINT4,
            activation_quantization_mode=None,
            activation_precision=jnp.float32,
        ),
    ]

    for config in configs:
        layer = config.random_init(4, (4,), has_biases=False, key=jax.random.key(0))
        layer = replace(layer, weights=jnp.full_like(layer.weights, 1.25))

        def loss(weights: jax.Array, *, current_layer: eqx.Module = layer) -> jax.Array:
            return replace(current_layer, weights=weights)(input_vector)[0].sum()

        grad = jax.grad(loss)(layer.weights)

        assert grad.shape == layer.weights.shape
        assert jnp.any(grad != 0)
