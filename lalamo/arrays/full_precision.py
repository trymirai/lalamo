from jaxtyping import Array, DTypeLike, Float, Key

from .base import ArrayForwardPassConfig, CompressedArray


class FullPrecisionArray(CompressedArray):
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

    @classmethod
    def compress(cls, weights: Float[Array, "... out_channels in_channels"]) -> "FullPrecisionArray":
        return cls(weights=weights)
