from enum import StrEnum

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike


class SScaleAxis(StrEnum):
    ROW = "row"
    COLUMN = "column"


def apply_post_gains(
    weights: Array, axes: tuple[SScaleAxis, ...], gains: tuple[Array, ...], dtype: DTypeLike
) -> Array:
    if not axes:
        return weights.astype(dtype)
    result = weights.astype(jnp.bfloat16)
    for axis, gain in zip(axes, gains, strict=True):
        factor = gain[..., None] if axis == SScaleAxis.ROW else gain
        # Each QAT fold rounds the weights before the next multiplication.
        result = jax.lax.optimization_barrier((result.astype(jnp.float32) * factor).astype(jnp.bfloat16))
    return result.astype(dtype)
