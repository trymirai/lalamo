from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import equinox as eqx
import jax.random as jr

from lalamo.common import is_abstract_array

from .base import ArrayForwardPassConfig, CompressedArray, StochasticQuantize
from .lora import LoraArray

if TYPE_CHECKING:
    from jaxtyping import Array, PRNGKeyArray


class CompositeArray(CompressedArray):
    parts: tuple[CompressedArray, ...] = eqx.field(default=())

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:  # noqa: B008
        match forward_pass_config.quantize:
            case StochasticQuantize(key=key):
                subkeys = jr.split(key, len(self.parts))
                first = self.parts[0].materialize(
                    replace(forward_pass_config, quantize=StochasticQuantize(key=subkeys[0]))
                )
                if is_abstract_array(first):
                    return first
                return sum(
                    (
                        part.materialize(replace(forward_pass_config, quantize=StochasticQuantize(key=subkey)))
                        for part, subkey in zip(self.parts, subkeys, strict=True)
                    )
                )
            case _:
                first = self.parts[0].materialize(forward_pass_config)
                if is_abstract_array(first):
                    return first
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
