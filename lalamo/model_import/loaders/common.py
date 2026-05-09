from jaxtyping import Array, Float

from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, ShapeDtypeMatrix, WeightMatrix

__all__ = [
    "load_full_precision",
]


def load_full_precision(
    template: WeightMatrix,
    weights: Float[Array, "*components out_channels in_channels"],
) -> FullPrecisionMatrix:
    if not isinstance(template, ShapeDtypeMatrix):
        raise TypeError(f"Expected ShapeDtypeMatrix, got {type(template).__name__}.")
    return FullPrecisionSpec(layout=template.spec.layout).compress(weights.astype(template.dtype))
