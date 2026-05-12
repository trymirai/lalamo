import jax
import jax.numpy as jnp
import pytest

from lalamo.compressed.awq import AWQSpec
from lalamo.compressed.e8p import E8PSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.compressed.mxfp4 import MXFP4Spec
from lalamo.compressed.normal_float import NormalFloatSpec
from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.weight_matrix import CompressionImplementation, Layout
from tests.common import assert_close_arrays


@pytest.mark.parametrize(
    ("spec", "expected_rate"),
    [
        (AWQSpec(bits=4, group_size=2), 14.0),
        (AWQSpec(bits=4, group_size=2, is_symmetric=True), 12.0),
        (MLXSpec(bits=4, group_size=2), 20.0),
        (NormalFloatSpec(bits=4, group_size=2), 8.0),
        (MXFP4Spec(group_size=2), 8.0),
        (E8PSpec(bits=2), 2.0),
        (E8PSpec(bits=3), 3.0),
        (E8PSpec(bits=4), 4.0),
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


@pytest.mark.parametrize(
    "spec",
    [
        AWQSpec(bits=4, group_size=4, layout=Layout.OUTPUT_INPUT),
        AWQSpec(bits=4, group_size=4, is_symmetric=True, layout=Layout.OUTPUT_INPUT),
        AWQSpec(bits=8, group_size=4, layout=Layout.INPUT_OUTPUT),
        AWQSpec(bits=8, group_size=4, is_symmetric=True, layout=Layout.INPUT_OUTPUT),
        MLXSpec(bits=4, group_size=4, layout=Layout.OUTPUT_INPUT),
        MLXSpec(bits=8, group_size=4, layout=Layout.INPUT_OUTPUT),
        NormalFloatSpec(bits=4, group_size=4, layout=Layout.OUTPUT_INPUT),
        NormalFloatSpec(bits=3, group_size=4, preserve_zero=False, layout=Layout.INPUT_OUTPUT),
        MXFP4Spec(group_size=4, layout=Layout.OUTPUT_INPUT),
        MXFP4Spec(group_size=4, layout=Layout.INPUT_OUTPUT),
        E8PSpec(bits=2, layout=Layout.OUTPUT_INPUT),
        E8PSpec(bits=3, layout=Layout.INPUT_OUTPUT),
        E8PSpec(bits=4, layout=Layout.OUTPUT_INPUT),
    ],
)
def test_quantized_spec_quantize_block_matches_training_compression(spec: QuantizedSpec) -> None:
    block_shape = (spec.output_block_size, spec.input_block_size)
    weights = (
        jnp.arange(spec.output_block_size * spec.input_block_size, dtype=jnp.float32).reshape(block_shape) - 3
    ) / 5

    expected = spec.compress(
        weights,
        implementation=CompressionImplementation.TRAINING,
        is_sharded=False,
    ).decompress()

    assert_close_arrays(result=spec.quantize_block(weights), reference=expected)


def test_quantized_spec_quantize_block_rejects_unexpected_shape() -> None:
    spec = AWQSpec(bits=4, group_size=4)
    weights = jnp.ones((spec.output_block_size + 1, spec.input_block_size), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Expected quantization block shape"):
        spec.quantize_block(weights)
