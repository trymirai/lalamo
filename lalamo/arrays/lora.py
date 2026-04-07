from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.modules.common import Initializer

from .base import ArrayForwardPassConfig, CompressedArray


class LoRAArray(CompressedArray, kind="lora"):
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
        key: PRNGKeyArray | None,  # noqa: ARG002
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
