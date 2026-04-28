from collections.abc import Iterator
from contextlib import contextmanager

from jax import default_matmul_precision
from jax.lax import DotAlgorithmPreset

__all__ = ["use_dot_algorithm_preset"]


def _dot_algorithm_preset_name(precision: DotAlgorithmPreset) -> str:
    if precision == DotAlgorithmPreset.DEFAULT:
        return "default"
    return precision.name


@contextmanager
def use_dot_algorithm_preset(precision: DotAlgorithmPreset) -> Iterator[None]:
    with default_matmul_precision(_dot_algorithm_preset_name(precision)):
        yield
