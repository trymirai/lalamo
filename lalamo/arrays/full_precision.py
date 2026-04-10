from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.common import ParameterPath

from .base import ArrayForwardPassConfig, CompressedArray, CompressedArraySpec

if TYPE_CHECKING:
    from lalamo.modules.common import Initializer


@dataclass(frozen=True)
class FullPrecisionSpec(CompressedArraySpec):
    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> "FullPrecisionArray":
        return FullPrecisionArray(spec=self, weights=weights)

    def init(
        self,
        initializer: "Initializer",
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> "FullPrecisionArray":
        return FullPrecisionArray(
            spec=self,
            weights=initializer.normal(1.0, (*leading_dims, out_channels, in_channels), initializer.precision),
        )

    def from_uzu(self, data: Mapping[str, Any], prefix: ParameterPath) -> "FullPrecisionArray":
        return FullPrecisionArray(spec=self, weights=data[prefix / "weights"])


class FullPrecisionArray(CompressedArray[FullPrecisionSpec]):
    weights: Float[Array, "... out_channels in_channels"]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def dequantize(
        self, quantized_weights: Float[Array, "... out_channels in_channels"]
    ) -> Float[Array, "... out_channels in_channels"]:
        return quantized_weights

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        return self.weights

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: Key[Array, ""] | None,  # noqa: ARG002
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: ARG002, B008
    ) -> Float[Array, "... out_channels"]:
        return self.weights @ vector
