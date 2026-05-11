import jax
import jax.numpy as jnp
import pytest

from lalamo.compressed.awq import AWQSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.weight_matrix import CompressionImplementation


@pytest.mark.parametrize(
    ("spec", "expected_rate"),
    [
        (AWQSpec(bits=4, group_size=2), 14.0),
        (AWQSpec(bits=4, group_size=2, is_symmetric=True), 12.0),
        (MLXSpec(bits=4, group_size=2), 20.0),
    ],
)
def test_quantized_spec_rate_counts_bits_per_weight_including_group_parameters(
    spec: QuantizedSpec,
    expected_rate: float,
) -> None:
    assert spec.rate == expected_rate


@pytest.mark.parametrize("spec", [AWQSpec(bits=4, group_size=32), MLXSpec(bits=4, group_size=32)])
def test_quantized_spec_distortion_matches_empirical_quantization_error(spec: QuantizedSpec) -> None:
    weights = jax.random.normal(jax.random.key(0), (256, 256), dtype=jnp.float32)
    compressed = spec.compress(weights, implementation=CompressionImplementation.TRAINING)
    empirical_mse = float(jnp.mean(jnp.square(weights - compressed.decompress())))
    assert empirical_mse == pytest.approx(spec.distortion, rel=0.15)
