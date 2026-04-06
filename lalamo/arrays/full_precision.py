from collections.abc import Mapping
from typing import Any

from jaxtyping import Array, Float

from lalamo.modules.common import Initializer

from .base import ArrayForwardPassConfig, CompressedArray


class FullPrecisionArray(CompressedArray, kind="full_precision"):
    weights: Float[Array, "... out_channels in_channels"]

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        return self.weights

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: ARG002, B008
    ) -> Float[Array, "... out_channels"]:
        return self.weights @ vector

    @classmethod
    def compress(cls, weights: Float[Array, "... out_channels in_channels"]) -> "FullPrecisionArray":
        return cls(weights=weights)

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> "FullPrecisionArray":
        return cls(weights=data["weights"])

    @classmethod
    def init(
        cls,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> "FullPrecisionArray":
        return cls(
            weights=initializer.normal(
                1.0,
                (*leading_dims, out_channels, in_channels),
                initializer.precision,
            )
        )
