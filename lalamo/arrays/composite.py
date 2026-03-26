from __future__ import annotations

import warnings
from functools import reduce

import equinox as eqx
import jax.core
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import quax
from lalamo.common import ParameterTree

from .base import QuantArray
from .lora import LoraArray


class CompositeArray(quax.Value):
    base: QuantArray
    loras: tuple[LoraArray, ...] = eqx.field(default=())

    def aval(self) -> jax.core.ShapedArray:
        return self.base.aval()

    def materialise(self) -> Array:
        warnings.warn("CompositeArray.materialise() — full materialization.", stacklevel=2)
        return self.value

    @property
    def value(self) -> Array:
        return reduce(lambda acc, lora: acc + lora.value, self.loras, self.base.value)

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
        result: dict[str, ParameterTree] = self.base.export_weights()  # type: ignore
        for i, lora in enumerate(self.loras):
            result[f"lora_{i}"] = lora.export_weights()
        return result

    @staticmethod
    def from_quant(base: QuantArray) -> CompositeArray:
        return CompositeArray(base=base)

    def add_lora(self, *, rank: int, scale: float = 1.0, key: PRNGKeyArray) -> CompositeArray:
        *_, out_channels, in_channels = self.shape
        lora = LoraArray.from_rank(out_channels, in_channels, rank=rank, scale=scale, key=key)
        return CompositeArray(base=self.base, loras=(*self.loras, lora))
