from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree

from .common import StateLayerBase

__all__ = ["DeltaNetStateLayer"]


class DeltaNetStateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch tokens conv_channels"]
    recurrent_state: Float[Array, "*batch heads head_k_dim head_v_dim"]

    def __post_init__(self) -> None:
        if self.conv_state.ndim not in (2, 3):
            raise ValueError(
                "Conv state must have 2 or 3 dimensions: [batch], tokens, conv_channels,"
                f" got shape {self.conv_state.shape}",
            )
        if self.recurrent_state.ndim not in (3, 4):
            raise ValueError(
                "Recurrent state must have 3 or 4 dimensions: [batch], heads, head_k_dim, head_v_dim,"
                f" got shape {self.recurrent_state.shape}",
            )

    @classmethod
    def init(
        cls,
        kernel_size: int,
        conv_dim: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        dtype: DTypeLike,
    ) -> Self:
        conv_state = jnp.zeros((kernel_size - 1, conv_dim), dtype=dtype)
        recurrent_state = jnp.zeros((num_v_heads, head_k_dim, head_v_dim), dtype=dtype)
        return cls(conv_state=conv_state, recurrent_state=recurrent_state)

    def export(self) -> ParameterTree:
        return dict(
            conv_state=self.conv_state,
            recurrent_state=self.recurrent_state,
        )
