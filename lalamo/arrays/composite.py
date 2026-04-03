from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.common import ParameterTree

from .base import ArrayForwardPassConfig, CompressedArray, StochasticQuantize
from .lora import LoraArray


class CompositeArray(CompressedArray):
    parts: tuple[CompressedArray, ...] = eqx.field(default=())

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:
        match forward_pass_config.quantize:
            case StochasticQuantize(key=key):
                subkeys = jr.split(key, len(self.parts))
                return sum(
                    part.materialize(replace(forward_pass_config, quantize=StochasticQuantize(key=subkey)))
                    for part, subkey in zip(self.parts, subkeys)
                )
            case _:
                return sum(part.materialize(forward_pass_config) for part in self.parts)

    def export_weights(self) -> ParameterTree:
        return {f"part_{i}": part.export_weights() for i, part in enumerate(self.parts)}

    @staticmethod
    def from_compressed(base: CompressedArray) -> CompositeArray:
        return CompositeArray(parts=(base,))

    def add_part(self, part: CompressedArray) -> CompositeArray:
        return CompositeArray(parts=(*self.parts, part))

    def add_lora(self, *, rank: int, key: PRNGKeyArray) -> CompositeArray:
        *_, out_channels, in_channels = self.materialize().shape
        lora = LoraArray.from_rank(out_channels, in_channels, rank=rank, key=key)
        return self.add_part(lora)
