from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.common import ParameterPath
from lalamo.modules.common import Initializer

from .base import ArrayForwardPassConfig, CompressedArray, CompressedArraySpec


@dataclass(frozen=True)
class LoRASpec(CompressedArraySpec):
    rank: int

    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> CompressedArray:
        raise NotImplementedError("LoRA does not support compression from full-precision weights")

    def init(
        self,
        initializer: "Initializer",
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> "LoRAArray":
        return LoRAArray(
            spec=self,
            down=initializer.zeros((*leading_dims, out_channels, self.rank), initializer.precision),
            up=initializer.zeros((*leading_dims, self.rank, in_channels), initializer.precision),
        )

    def from_uzu(self, data: Mapping[str, Any], prefix: ParameterPath) -> "LoRAArray":
        return LoRAArray(spec=self, down=data[prefix / "down"], up=data[prefix / "up"])


class LoRAArray(CompressedArray[LoRASpec]):
    down: Float[Array, "... out_channels rank"]
    up: Float[Array, "... rank in_channels"]

    @property
    def shape(self) -> tuple[int, ...]:
        *leading, out_channels, _rank = self.down.shape
        *_, in_channels = self.up.shape
        return (*leading, out_channels, in_channels)

    @property
    def dtype(self) -> DTypeLike:
        return self.down.dtype

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        return self.down @ self.up

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: Key[Array, ""],  # noqa: ARG002
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: ARG002, B008
    ) -> Float[Array, " out_channels"]:
        return self.down @ (self.up @ vector)
