from __future__ import annotations

from collections.abc import Mapping

import jax.core
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree, dummy_array

from .base import CompressedArray


class FullPrecisionArray(CompressedArray):
    data: Float[Array, "... out_channels in_channels"]

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.data.shape, self.data.dtype)

    def dot(self, vector: Float[Array, " in_channels"]) -> Float[Array, " out_channels"]:
        return self.data @ vector

    @property
    def value(self) -> Array:
        return self.data

    def export_weights(self) -> ParameterTree:
        return dict(weights=self.data)

    @staticmethod
    def from_array(weights: Array) -> FullPrecisionArray:
        return FullPrecisionArray(data=weights)

    @staticmethod
    def empty(
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
        precision: DTypeLike,
    ) -> FullPrecisionArray:
        return FullPrecisionArray(data=dummy_array((*leading_dims, out_channels, in_channels), precision))

    @staticmethod
    def import_weights(weights_map: Mapping[str, Array]) -> FullPrecisionArray:
        return FullPrecisionArray(data=weights_map["weights"])

    @staticmethod
    def from_torch(
        state_dict: Mapping[str, Array],
        *,
        prefix: str,
        dtype: DTypeLike,
    ) -> FullPrecisionArray:
        return FullPrecisionArray(data=state_dict[f"{prefix}.weight"].astype(dtype))
