from collections.abc import Mapping
from typing import Any

from jaxtyping import Array, Float

from lalamo.modules.common import Initializer

from .base import ArrayForwardPassConfig, CompressedArray


class LoRAArray(CompressedArray, kind="lora"):
    down: Float[Array, "... out_channels rank"]
    up: Float[Array, "... rank in_channels"]

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        return self.down @ self.up

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: ARG002, B008
    ) -> Float[Array, " out_channels"]:
        return self.down @ (self.up @ vector)

    @classmethod
    def init(
        cls,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
        *,
        rank: int,
    ) -> "LoRAArray":
        return cls(
            down=initializer.zeros((*leading_dims, out_channels, rank), initializer.precision),
            up=initializer.zeros((*leading_dims, rank, in_channels), initializer.precision),
        )
