from jaxtyping import Array, Float

from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, ShapeDtypeMatrix

__all__ = [
    "load_full_precision",
]


def load_full_precision(
    template: ShapeDtypeMatrix,
    weights: Float[Array, "... out_channels in_channels"],
) -> FullPrecisionMatrix:
    assert isinstance(template, ShapeDtypeMatrix)
    return FullPrecisionSpec(layout=template.spec.layout).compress(weights.astype(template.dtype))
