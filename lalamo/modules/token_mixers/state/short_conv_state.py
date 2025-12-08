from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree

from .common import StateLayerBase

__all__ = ["ShortConvStateLayer"]


class ShortConvStateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch tokens conv_channels"]

    def __post_init__(self) -> None:
        if self.conv_state.ndim not in (2, 3):
            raise ValueError(
                f"Conv state must have 2 or 3 dimensions: [batch], tokens, conv_channels,"
                f" got shape {self.conv_state.shape}",
            )

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
