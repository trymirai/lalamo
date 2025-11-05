import os
import sys

sys.path.append(os.path.abspath(os.getcwd() + os.path.sep + ".."))

import jax.numpy as jnp
from jax import random

from lalamo.modules.normalization import NormalizationConfig, UpcastMode


def test_rmsnorm_without_subtract_mean():
    """Test standard RMSNorm behavior (subtract_mean=False)."""
    rng_key = random.PRNGKey(42)
    input_dim = 128
    test_input = random.normal(rng_key, (input_dim,))

    config = NormalizationConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    norm = config.init(input_dim)
    output = norm(test_input)

    # Verify output shape
    assert output.shape == test_input.shape

    # Verify RMS normalization: RMS of output should be close to 1
    # (when scales are 1 and no offset)
    rms = jnp.sqrt(jnp.mean(jnp.square(output)))
    assert jnp.isclose(rms, 1.0, rtol=1e-4)


def test_rmsnorm_with_subtract_mean():
    """Test RMSNorm with subtract_mean=True (LayerNorm-like behavior)."""
    rng_key = random.PRNGKey(42)
    input_dim = 128
    test_input = random.normal(rng_key, (input_dim,))

    config = NormalizationConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=True,
    )
    norm = config.init(input_dim)
    output = norm(test_input)

    # Verify output shape
    assert output.shape == test_input.shape

    # Verify mean is close to 0 (mean subtraction occurred)
    output_mean = jnp.mean(output)
    assert jnp.isclose(output_mean, 0.0, atol=1e-5)

    # Verify variance is close to 1 (when scales are 1)
    output_var = jnp.var(output)
    assert jnp.isclose(output_var, 1.0, rtol=1e-4)


def test_subtract_mean_changes_output():
    """Verify that subtract_mean parameter actually changes the output."""
    rng_key = random.PRNGKey(42)
    input_dim = 128
    test_input = random.normal(rng_key, (input_dim,))

    config_without = NormalizationConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    config_with = NormalizationConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=True,
    )

    norm_without = config_without.init(input_dim)
    norm_with = config_with.init(input_dim)

    output_without = norm_without(test_input)
    output_with = norm_with(test_input)

    # Outputs should be different
    assert not jnp.allclose(output_without, output_with)


def test_rmsnorm_with_custom_scales():
    """Test RMSNorm with custom scale values."""
    rng_key = random.PRNGKey(42)
    input_dim = 128
    test_input = random.normal(rng_key, (input_dim,))

    config = NormalizationConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=True,
    )
    norm = config.init(input_dim)

    # Modify scales to be non-uniform
    custom_scales = jnp.linspace(0.5, 1.5, input_dim)
    from dataclasses import replace

    norm = replace(norm, scales=custom_scales)

    output = norm(test_input)

    # Verify output shape
    assert output.shape == test_input.shape

    # Mean should still be close to 0 due to subtract_mean
    output_mean = jnp.mean(output)
    assert jnp.isclose(
        output_mean, 0.0, atol=0.1
    )  # Looser tolerance with non-uniform scales


def test_rmsnorm_zero_input():
    """Test RMSNorm with zero input."""
    input_dim = 128
    zero_input = jnp.zeros(input_dim)

    for subtract_mean in [False, True]:
        config = NormalizationConfig(
            scale_precision=jnp.float32,
            accumulation_precision=jnp.float32,
            epsilon=1e-5,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=subtract_mean,
        )
        norm = config.init(input_dim)
        output = norm(zero_input)

        # Output should be zero (or very close to it)
        assert jnp.allclose(output, 0.0, atol=1e-6)


def test_rmsnorm_vs_torch_layernorm():
    """
    Compare RMSNorm with subtract_mean=True against PyTorch's LayerNorm.
    They should produce equivalent outputs.
    """
    import torch
    import numpy as np

    rng_key = random.PRNGKey(42)
    input_dim = 128
    test_input_jax = random.normal(rng_key, (input_dim,))

    # Convert to numpy and then to torch
    test_input_np = np.array(test_input_jax)
    test_input_torch = torch.from_numpy(test_input_np).float()

    # RMSNorm with subtract_mean=True
    config = NormalizationConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=True,
    )
    rmsnorm = config.init(input_dim)
    rmsnorm_output = rmsnorm(test_input_jax)

    # PyTorch LayerNorm
    layernorm = torch.nn.LayerNorm(input_dim, eps=1e-5, elementwise_affine=True)
    # Initialize weights to ones (same as RMSNorm)
    torch.nn.init.ones_(layernorm.weight)
    torch.nn.init.zeros_(layernorm.bias)

    with torch.no_grad():
        layernorm_output = layernorm(test_input_torch)

    # Convert back to numpy for comparison
    layernorm_output_np = layernorm_output.numpy()
    rmsnorm_output_np = np.array(rmsnorm_output)

    # They should be very close
    assert np.allclose(rmsnorm_output_np, layernorm_output_np, rtol=1e-5, atol=1e-5)
