import jax
import jax.numpy as jnp
import pytest

from lalamo.compressed.int import IntSpec
from lalamo.compressed.lloyd_max import LloydMaxSpec
from lalamo.compressed.microfloat import MicrofloatScaleMode, MicrofloatSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.weight_matrix import CompressionImplementation, Layout
from tests.common import assert_close_arrays
from tests.helpers import make_test_sharding_config


@pytest.mark.parametrize(
    "spec",
    [
        IntSpec(bits=4, group_size=32),
        IntSpec(bits=8, group_size=32),
        IntSpec(bits=4, group_size=32, is_symmetric=True),
        MLXSpec(bits=4, group_size=32),
        MLXSpec(bits=8, group_size=32),
        LloydMaxSpec(bits=3, group_size=32),
        LloydMaxSpec(bits=4, group_size=32, bias_bits=3),
        LloydMaxSpec(bits=6, group_size=32, bias_bits=6),
        MicrofloatSpec(group_size=32),
        MicrofloatSpec(group_size=16, scale_mode=MicrofloatScaleMode.NVFP4),
    ],
)
def test_quantized_spec_rate_matches_compressed_array_byte_count(spec: QuantizedSpec) -> None:
    weights = jax.random.normal(jax.random.key(1), (64, 512), dtype=jnp.float32)
    compressed = spec.compress(
        weights, implementation=CompressionImplementation.INFERENCE, sharding_config=make_test_sharding_config()
    )
    byte_count = sum(array.size * array.dtype.itemsize for array in compressed.export().arrays.values())
    empirical_rate = 8 * byte_count / weights.size

    assert empirical_rate == pytest.approx(spec.rate, abs=0.002)


@pytest.mark.parametrize("scale_mode", list(MicrofloatScaleMode))
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
def test_microfloat_rate_matches_compressed_array_byte_count(
    scale_mode: MicrofloatScaleMode,
    group_size: int,
) -> None:
    spec = MicrofloatSpec(group_size=group_size, scale_mode=scale_mode)
    weights = jax.random.normal(jax.random.key(1), (256, 512), dtype=jnp.float32)
    compressed = spec.compress(
        weights, implementation=CompressionImplementation.INFERENCE, sharding_config=make_test_sharding_config()
    )
    byte_count = sum(array.size * array.dtype.itemsize for array in compressed.export().arrays.values())
    empirical_rate = 8 * byte_count / weights.size

    assert empirical_rate == pytest.approx(spec.rate, abs=0.001)


@pytest.mark.parametrize(
    "spec",
    [
        IntSpec(bits=4, group_size=32),
        IntSpec(bits=4, group_size=32, is_symmetric=True),
        MLXSpec(bits=4, group_size=32),
        LloydMaxSpec(bits=4, group_size=32),
        LloydMaxSpec(bits=4, group_size=32, bias_bits=4),
        MicrofloatSpec(group_size=32),
        MicrofloatSpec(group_size=16, scale_mode=MicrofloatScaleMode.NVFP4),
    ],
)
@pytest.mark.slow
def test_quantized_spec_distortion_matches_monte_carlo_quantization_error(spec: QuantizedSpec) -> None:
    weights = jax.random.normal(jax.random.key(2), (512, 512), dtype=jnp.float32)
    compressed = spec.compress(
        weights, implementation=CompressionImplementation.TRAINING, sharding_config=make_test_sharding_config()
    )
    empirical_mse = float(jnp.mean(jnp.square(weights - compressed.decompress())))

    assert empirical_mse == pytest.approx(spec.distortion, rel=0.15)


@pytest.mark.parametrize("scale_mode", list(MicrofloatScaleMode))
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
@pytest.mark.slow
def test_microfloat_distortion_matches_inference_quantization_error(
    scale_mode: MicrofloatScaleMode,
    group_size: int,
) -> None:
    spec = MicrofloatSpec(group_size=group_size, scale_mode=scale_mode)
    weights = jax.random.normal(jax.random.key(2), (512, 512), dtype=jnp.float32)
    compressed = spec.compress(
        weights, implementation=CompressionImplementation.INFERENCE, sharding_config=make_test_sharding_config()
    )
    empirical_mse = float(jnp.mean(jnp.square(weights - compressed.decompress())))

    assert empirical_mse == pytest.approx(spec.distortion, rel=0.15)


@pytest.mark.parametrize(
    "spec",
    [
        IntSpec(bits=4, group_size=4, layout=Layout.OUTPUT_INPUT),
        IntSpec(bits=4, group_size=4, is_symmetric=True, layout=Layout.OUTPUT_INPUT),
        IntSpec(bits=8, group_size=4, layout=Layout.INPUT_OUTPUT),
        IntSpec(bits=8, group_size=4, is_symmetric=True, layout=Layout.INPUT_OUTPUT),
        MLXSpec(bits=4, group_size=4, layout=Layout.OUTPUT_INPUT),
        MLXSpec(bits=8, group_size=4, layout=Layout.INPUT_OUTPUT),
        LloydMaxSpec(bits=4, group_size=4, layout=Layout.OUTPUT_INPUT),
        LloydMaxSpec(bits=3, group_size=4, layout=Layout.INPUT_OUTPUT),
        MicrofloatSpec(group_size=4, layout=Layout.OUTPUT_INPUT),
        MicrofloatSpec(group_size=4, layout=Layout.INPUT_OUTPUT),
    ],
)
def test_quantized_spec_quantize_block_matches_inference_compression(spec: QuantizedSpec) -> None:
    block_shape = (spec.output_block_size, spec.input_block_size)
    weights = (
        jnp.arange(spec.output_block_size * spec.input_block_size, dtype=jnp.float32).reshape(block_shape) - 3
    ) / 5

    expected = spec.compress(
        weights,
        implementation=CompressionImplementation.INFERENCE,
        sharding_config=make_test_sharding_config().replicated_with_same_mesh(),
    ).decompress()

    assert_close_arrays(
        result=spec.quantize_block(weights, sharding_config=make_test_sharding_config()),
        reference=expected,
    )


def test_quantized_spec_quantize_block_supports_batched_blocks() -> None:
    spec = IntSpec(bits=4, group_size=4)
    weights = jnp.stack(
        [
            jnp.linspace(-1, 1, spec.input_block_size, dtype=jnp.float32).reshape(
                spec.output_block_size,
                spec.input_block_size,
            ),
            jnp.linspace(1, -1, spec.input_block_size, dtype=jnp.float32).reshape(
                spec.output_block_size,
                spec.input_block_size,
            ),
        ],
    )

    expected = spec.compress(
        weights,
        implementation=CompressionImplementation.INFERENCE,
        sharding_config=make_test_sharding_config().replicated_with_same_mesh(),
    ).decompress()

    assert_close_arrays(
        result=spec.quantize_block(weights, sharding_config=make_test_sharding_config()),
        reference=expected,
    )


def test_quantized_spec_quantize_block_rejects_unexpected_shape() -> None:
    spec = IntSpec(bits=4, group_size=4)
    weights = jnp.ones((spec.output_block_size + 1, spec.input_block_size), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Expected quantization block shape"):
        spec.quantize_block(weights, sharding_config=make_test_sharding_config())
