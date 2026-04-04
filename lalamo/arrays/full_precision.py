from __future__ import annotations

from typing import TYPE_CHECKING

from lalamo.common import is_abstract_array

from .base import ArrayForwardPassConfig, CompressedArray

if TYPE_CHECKING:
    from jaxtyping import Array, DTypeLike, Float

    from lalamo.modules.common import Initializer


class FullPrecisionArray(CompressedArray):
    raw: Float[Array, "... out_channels in_channels"]

    @property
    def is_abstract(self) -> bool:
        return is_abstract_array(self.raw)

    def materialize(self, forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig()) -> Array:  # noqa: ARG002, B008
        return self.raw

    @classmethod
    def init(
        cls,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> FullPrecisionArray:
        return cls(raw=initializer.zeros((*leading_dims, out_channels, in_channels), initializer.precision))

    @classmethod
    def from_weight(
        cls,
        weights: Array,
        *,
        dtype: DTypeLike,
    ) -> FullPrecisionArray:
        return cls(raw=weights.astype(dtype))
