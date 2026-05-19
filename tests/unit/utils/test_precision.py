from typing import cast

import jax
import pytest
from jax.lax import DotAlgorithmPreset

from lalamo.utils.precision import use_dot_algorithm_preset


def _current_matmul_precision() -> str | None:
    return cast("str | None", object.__getattribute__(jax.config, "jax_default_matmul_precision"))


@pytest.mark.parametrize(
    ("precision", "config_value"),
    [
        (DotAlgorithmPreset.DEFAULT, "default"),
        (DotAlgorithmPreset.F16_F16_F32, "F16_F16_F32"),
        (DotAlgorithmPreset.BF16_BF16_F32, "BF16_BF16_F32"),
        (DotAlgorithmPreset.F32_F32_F32, "F32_F32_F32"),
    ],
)
def test_use_dot_algorithm_preset_sets_jax_config_value(
    precision: DotAlgorithmPreset,
    config_value: str,
) -> None:
    previous_value = _current_matmul_precision()

    with use_dot_algorithm_preset(precision):
        assert _current_matmul_precision() == config_value

    assert _current_matmul_precision() == previous_value


@pytest.mark.parametrize(
    ("precision", "config_value"),
    [
        (DotAlgorithmPreset.DEFAULT, "default"),
        (DotAlgorithmPreset.F16_F16_F32, "F16_F16_F32"),
        (DotAlgorithmPreset.BF16_BF16_F32, "BF16_BF16_F32"),
        (DotAlgorithmPreset.F32_F32_F32, "F32_F32_F32"),
    ],
)
def test_use_dot_algorithm_preset_can_be_used_as_decorator(
    precision: DotAlgorithmPreset,
    config_value: str,
) -> None:
    previous_value = _current_matmul_precision()

    @use_dot_algorithm_preset(precision)
    def current_matmul_precision() -> str | None:
        return _current_matmul_precision()

    assert current_matmul_precision() == config_value
    assert _current_matmul_precision() == previous_value
