from collections.abc import Mapping
from typing import Any

from jaxtyping import Array, Float

from lalamo.serialization import strip_uzu_prefix

from .base import ArrayForwardPassConfig, CompressedArray


class CompositeArray(CompressedArray, kind="composite"):
    parts: tuple[CompressedArray, ...]

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        assert len(self.parts) != 0, "CompositeArray must be non-empty to materialize"
        return sum(part.materialize() for part in self.parts)

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, "... out_channels"]:
        assert len(self.parts) != 0, "CompositeArray must be non-empty to dot"
        return sum(part.dot(vector, forward_pass_config) for part in self.parts)

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> "CompositeArray":
        parts = []
        i = 0
        while f"parts.{i}.__kind__" in data:
            parts.append(CompressedArray.from_uzu(strip_uzu_prefix(data, f"parts.{i}")))
            i += 1
        return cls(parts=tuple(parts))

    def add_part(self, part: CompressedArray) -> "CompositeArray":
        return CompositeArray(parts=(*self.parts, part))
