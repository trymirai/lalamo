from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import equinox as eqx
import jax.random as jr

from .base import ArrayForwardPassConfig, CompressedArray, StochasticQuantize
from .lora import LoraArray

if TYPE_CHECKING:
    from jaxtyping import Array, PRNGKeyArray


class CompositeArray(CompressedArray):
    parts: tuple[CompressedArray, ...] = eqx.field(default=())

    @property
    def is_abstract(self) -> bool:
        return self.parts[0].is_abstract

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:  # noqa: B008
        match forward_pass_config.quantize:
            case StochasticQuantize(key=key):
                part_configs = tuple(
                    replace(forward_pass_config, quantize=StochasticQuantize(key=subkey))
                    for subkey in jr.split(key, len(self.parts))
                )
            case _:
                part_configs = (forward_pass_config,) * len(self.parts)

        materialized_parts = [
            part.materialize(part_forward_pass_config)
            for part, part_forward_pass_config in zip(self.parts, part_configs, strict=True)
        ]
        first_part, *remaining_parts = materialized_parts
        if self.is_abstract:
            return first_part
        return sum(remaining_parts, start=first_part)

    def to_uzu(self) -> dict[str, Array]:
        result: dict[str, Array] = {}
        for index, part in enumerate(self.parts):
            part_name = "base" if index == 0 else "lora" if isinstance(part, LoraArray) else f"part_{index}"
            for key, value in part.to_uzu().items():
                result[f"{part_name}.{key}"] = value
        return result

    def from_uzu(self, weights: dict[str, Array]) -> CompositeArray:
        restored_parts: list[CompressedArray] = []
        for index, part in enumerate(self.parts):
            prefix = "base." if index == 0 else "lora." if isinstance(part, LoraArray) else f"part_{index}."
            part_weights = {
                key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)
            }
            restored_parts.append(part.from_uzu(part_weights))
        return type(self)(parts=tuple(restored_parts))

    @staticmethod
    def from_compressed(base: CompressedArray) -> CompositeArray:
        return CompositeArray(parts=(base,))

    def add_part(self, part: CompressedArray) -> CompositeArray:
        return CompositeArray(parts=(*self.parts, part))

    def add_lora(self, *, rank: int, key: PRNGKeyArray) -> CompositeArray:
        *_, out_channels, in_channels = self.materialize().shape
        lora = LoraArray.from_rank(out_channels, in_channels, rank=rank, key=key)
        return self.add_part(lora)
