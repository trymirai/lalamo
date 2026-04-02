from __future__ import annotations

import warnings
from functools import reduce

import equinox as eqx
import jax.core
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.common import ParameterTree

from .base import CompressedArray
from .lora import LoraArray


class CompositeArray(CompressedArray):
    parts: tuple[CompressedArray, ...] = eqx.field(default=())

    def aval(self) -> jax.core.ShapedArray:
        if not self.parts:
            raise ValueError("CompositeArray has no parts")
        return self.parts[0].aval()

    def dot(self, vector: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        return reduce(lambda acc, part: acc + part.dot(vector), self.parts[1:], self.parts[0].dot(vector))

    def materialise(self) -> Array:
        warnings.warn("CompositeArray.materialise() — full materialization.", stacklevel=2)
        return self.value

    @property
    def value(self) -> Array:
        return reduce(lambda acc, part: acc + part.value, self.parts[1:], self.parts[0].value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.aval().shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.aval().dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def export_weights(self) -> ParameterTree:
        result: dict[str, ParameterTree] = {}
        for i, part in enumerate(self.parts):
            result[f"part_{i}"] = part.export_weights()
        return result

    @staticmethod
    def from_compressed(base: CompressedArray) -> CompositeArray:
        return CompositeArray(parts=(base,))

    def add_part(self, part: CompressedArray) -> CompositeArray:
        return CompositeArray(parts=(*self.parts, part))

    def add_lora(self, *, rank: int, scale: float = 1.0, key: PRNGKeyArray) -> CompositeArray:
        *_, out_channels, in_channels = self.shape
        lora = LoraArray.from_rank(out_channels, in_channels, rank=rank, scale=scale, key=key)
        return self.add_part(lora)
