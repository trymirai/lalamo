import pytest

from lalamo.compressed.awq import AWQSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.compressed.utils.gaussian_order_statistics import (
    standard_normal_absmax_squared,
    standard_normal_range_squared,
)
from lalamo.weight_matrix import FullPrecisionSpec, Layout


def test_only_quantized_specs_expose_block_size_properties() -> None:
    assert isinstance(AWQSpec(bits=4, group_size=2), QuantizedSpec)
    assert isinstance(MLXSpec(bits=4, group_size=2), QuantizedSpec)
    assert not hasattr(FullPrecisionSpec(), "input_block_size")
    assert not hasattr(FullPrecisionSpec(), "output_block_size")


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


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_quantized_spec_distortion_is_layout_invariant_for_isotropic_gaussian(layout: Layout) -> None:
    assert AWQSpec(bits=4, group_size=2, layout=layout).distortion == pytest.approx(2 / (12 * 15**2))
    assert MLXSpec(bits=4, group_size=2, layout=layout).distortion == pytest.approx(2 / (12 * 15**2))


def test_symmetric_awq_distortion_uses_gaussian_absmax() -> None:
    assert AWQSpec(bits=4, group_size=1, is_symmetric=True).distortion == pytest.approx(1 / (12 * 7**2))


def test_standard_normal_order_statistic_results_are_cached() -> None:
    standard_normal_range_squared.cache_clear()
    standard_normal_absmax_squared.cache_clear()

    standard_normal_range_squared(4)
    standard_normal_range_squared(4)
    standard_normal_absmax_squared(4)
    standard_normal_absmax_squared(4)

    assert standard_normal_range_squared.cache_info().hits == 1
    assert standard_normal_absmax_squared.cache_info().hits == 1
