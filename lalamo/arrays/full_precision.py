from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterPath, ParameterTree, dummy_array

from .base import CompressedArray


class FullPrecisionArray(CompressedArray):
    raw: Float[Array, "... out_channels in_channels"]

    def materialize(self, forward_pass_config=None) -> Array:
        return self.raw

    def export_weights(self) -> ParameterTree:
        return dict(weights=self.raw)

    @staticmethod
    def empty(
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
        precision: DTypeLike,
    ) -> FullPrecisionArray:
        return FullPrecisionArray(raw=dummy_array((*leading_dims, out_channels, in_channels), precision))

    @staticmethod
    def import_weights(weights_map: Mapping[str, Array]) -> FullPrecisionArray:
        return FullPrecisionArray(raw=weights_map["weights"])

    @staticmethod
    def from_torch(
        state_dict: Mapping[str, Array],
        *,
        prefix: ParameterPath | str,
        dtype: DTypeLike,
    ) -> FullPrecisionArray:
        path = ParameterPath(prefix)
        return FullPrecisionArray(raw=state_dict[path / "weight"].astype(dtype))
