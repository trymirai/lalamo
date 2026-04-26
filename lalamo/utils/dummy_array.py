from typing import cast

from jax import ShapeDtypeStruct
from jax.sharding import Sharding
from jaxtyping import Array, DTypeLike

__all__ = [
    "dummy_array",
]


def dummy_array(shape: int | tuple[int, ...], dtype: DTypeLike, sharding: Sharding | None = None) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    return cast("Array", ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding))
