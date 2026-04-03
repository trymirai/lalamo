from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.common import ParameterTree

from .base import ArrayForwardPassConfig, CompressedArray
from .lora import LoraArray


class CompositeArray(CompressedArray):
    parts: tuple[CompressedArray, ...] = eqx.field(default=())

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:
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
