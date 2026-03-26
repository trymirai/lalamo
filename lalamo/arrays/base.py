from __future__ import annotations

import abc
import warnings

import jax.core
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array

import quax
from lalamo.common import ParameterTree


class QuantArray(quax.Value):
    @abc.abstractmethod
    def aval(self) -> jax.core.ShapedArray: ...

    @property
    @abc.abstractmethod
    def value(self) -> Array: ...

    @abc.abstractmethod
    def export_weights(self) -> ParameterTree: ...

    def materialise(self) -> Array:
        warnings.warn(
            f"{type(self).__name__}.materialise() — implicit dequantization fallback.",
            stacklevel=2,
        )
        return self.value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.aval().shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.aval().dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)


def unpack_int32(packed: Array, bits: int) -> Array:
    shifts = jnp.arange(0, 32, bits)
    mask = (2**bits) - 1
    unpacked = jnp.bitwise_and(jnp.right_shift(packed[:, :, None], shifts[None, None, :]), mask)
    return rearrange(unpacked, "rows packed_groups packed_vals -> rows (packed_groups packed_vals)")
