from jaxtyping import Array, DTypeLike, Float, Key

from .base import ArrayForwardPassConfig, CompressedArray


class LoRAArray(CompressedArray):
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
        key: Key[Array, ""] | None,  # noqa: ARG002
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: ARG002, B008
    ) -> Float[Array, " out_channels"]:
        return self.down @ (self.up @ vector)
