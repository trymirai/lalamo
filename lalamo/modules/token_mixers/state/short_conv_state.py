from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree

from .common import StateLayerBase

__all__ = ["ShortConvStateLayer"]


class ShortConvStateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch tokens conv_channels"]

    @classmethod
    def init(
        cls,
        kernel_size: int,
        model_dim: int,
        dtype: DTypeLike,
    ) -> Self:
        return cls(conv_state=jnp.zeros((kernel_size - 1, model_dim), dtype=dtype))

    def export(self) -> ParameterTree:
        return dict(conv_state=self.conv_state)
