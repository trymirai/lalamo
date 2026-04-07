from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.serialization import strip_uzu_prefix

from .base import ArrayForwardPassConfig, CompressedArray


class CompositeArray(CompressedArray, kind="composite"):
    parts: tuple[CompressedArray, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        assert len(self.parts) != 0, "CompositeArray must be non-empty"
        first, *rest = self.parts
        result = first.shape
        for part in rest:
            if part.shape != result:
                raise ValueError(
                    f"CompositeArray parts have mismatched shapes: {result} vs {part.shape}",
                )
        return result

    @property
    def dtype(self) -> DTypeLike:
        assert len(self.parts) != 0, "CompositeArray must be non-empty"
        return jnp.result_type(*(part.dtype for part in self.parts))

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        assert len(self.parts) != 0, "CompositeArray must be non-empty to materialize"
        return sum(part.materialize() for part in self.parts)

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: PRNGKeyArray | None,
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, "... out_channels"]:
        assert len(self.parts) != 0, "CompositeArray must be non-empty to dot"
        return sum(part.dot(vector, key=key, forward_pass_config=forward_pass_config) for part in self.parts)

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> CompressedArray:
        if str(data.get("__kind__")) != cls.kind:
            return CompressedArray.from_uzu(data)
        parts = []
        i = 0
        while f"parts.{i}.__kind__" in data:
            parts.append(CompressedArray.from_uzu(strip_uzu_prefix(data, f"parts.{i}")))
            i += 1
        return cls(parts=tuple(parts))

    def add_part(self, part: CompressedArray) -> "CompositeArray":
        return CompositeArray(parts=(*self.parts, part))
