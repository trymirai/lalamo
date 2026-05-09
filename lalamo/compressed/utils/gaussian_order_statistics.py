from functools import cache
from statistics import NormalDist

from numpy.polynomial.legendre import leggauss

__all__ = [
    "standard_normal_absmax_squared",
    "standard_normal_range_squared",
]

_GAUSS_LEGENDRE_ORDER = 100
_STANDARD_NORMAL = NormalDist()


@cache
def _unit_legendre_quadrature() -> tuple[tuple[float, ...], tuple[float, ...]]:
    nodes, weights = leggauss(_GAUSS_LEGENDRE_ORDER)
    unit_nodes = tuple(float((node + 1) / 2) for node in nodes)
    unit_weights = tuple(float(weight / 2) for weight in weights)
    return unit_nodes, unit_weights


@cache
def standard_normal_absmax_squared(group_size: int) -> float:
    if group_size == 1:
        return 1.0

    probabilities, weights = _unit_legendre_quadrature()
    total = sum(
        weight
        * _STANDARD_NORMAL.inv_cdf((probability + 1) / 2) ** 2
        * probability ** (group_size - 1)
        for probability, weight in zip(probabilities, weights, strict=True)
    )
    return group_size * total


@cache
def standard_normal_range_squared(group_size: int) -> float:
    if group_size == 1:
        return 0.0

    if group_size == 2:
        return 2.0

    probabilities, weights = _unit_legendre_quadrature()
    total = 0.0
    for min_probability, min_weight in zip(probabilities, weights, strict=True):
        min_value = _STANDARD_NORMAL.inv_cdf(min_probability)
        upper_probability_scale = 1 - min_probability
        min_probability_weight = upper_probability_scale ** (group_size - 1)

        conditional_total = 0.0
        for span_fraction, span_weight in zip(probabilities, weights, strict=True):
            max_probability = min_probability + upper_probability_scale * span_fraction
            value_range = _STANDARD_NORMAL.inv_cdf(max_probability) - min_value
            conditional_total += span_weight * value_range**2 * span_fraction ** (group_size - 2)

        total += min_weight * min_probability_weight * conditional_total

    return group_size * (group_size - 1) * total
