import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from lalamo.quantization.incoherence import hadamard_transform, random_incoherence_keys


@pytest.mark.parametrize("block_size", [2, 4, 8, 16])
@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_hadamard_round_trip(block_size: int, num_blocks: int) -> None:
    """Test that forward followed by inverse transform recovers the original input."""
    channels = block_size * num_blocks
    inputs = jnp.arange(channels, dtype=jnp.float32)

    transformed = hadamard_transform(inputs, block_size, inverse=False)
    recovered = hadamard_transform(transformed, block_size, inverse=True)

    assert jnp.allclose(recovered, inputs, atol=1e-5)


@pytest.mark.parametrize("block_size", [2, 4, 8, 16])
@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_hadamard_norm_preservation(block_size: int, num_blocks: int) -> None:
    """Test that the Hadamard transform preserves the norm of the input."""
    channels = block_size * num_blocks
    inputs = jnp.arange(channels, dtype=jnp.float32)

    transformed = hadamard_transform(inputs, block_size, inverse=False)

    input_norm = jnp.linalg.norm(inputs)
    output_norm = jnp.linalg.norm(transformed)

    assert jnp.allclose(input_norm, output_norm, atol=1e-5)


@pytest.mark.parametrize("block_size", [32, 64])
def test_hadamard_with_incoherence_normalizes_outliers(block_size: int) -> None:
    """Test that random incoherence keys + Hadamard transform produces approximately normal distribution."""
    num_channels = 4096

    # Create input with outliers: mostly small values with a few large ones
    key = jax.random.PRNGKey(42)
    inputs_key, incoherence_key = jax.random.split(key)

    # Generate data with outliers
    inputs = jax.random.normal(inputs_key, shape=(num_channels,)) * 0.1
    # Add some outliers (10% of values are 10x larger)
    outlier_indices = jnp.arange(0, num_channels, 10)
    inputs = inputs.at[outlier_indices].multiply(10.0)

    # Apply random incoherence transformation
    incoherence_keys = random_incoherence_keys(num_channels, 1, incoherence_key)
    inputs_with_incoherence = inputs * incoherence_keys.input_factors

    # Apply Hadamard transform
    transformed = hadamard_transform(inputs_with_incoherence, block_size, inverse=False)

    # Test for normality using Shapiro-Wilk test
    # p-value > 0.05 suggests data is approximately normal
    _, p_value = stats.shapiro(np.array(transformed))

    assert p_value > 0.05, f"Distribution not normal (p={p_value:.4f})"
