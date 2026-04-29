from collections.abc import Callable, Iterable

import equinox as eqx
from jaxtyping import Array, Float, PyTree

from lalamo.utils.surgery import load_as, load_as_at
from lalamo.weight_matrix import FullPrecisionMatrix, WeightMatrix

__all__ = [
    "load_full_precision",
    "load_parameters",
]


def load_full_precision(
    template: WeightMatrix,
    weights: Float[Array, "... out_channels in_channels"],
) -> FullPrecisionMatrix:
    assert isinstance(template, FullPrecisionMatrix)
    return load_as(template, template.spec.compress(weights), allow_dtype_cast=True)


def load_parameters[M: eqx.Module](
    selector: Callable[[M], Iterable[PyTree]],
    module: M,
    new_values: Iterable[PyTree],
) -> M:
    return load_as_at(selector, module, new_values, allow_dtype_cast=True)
