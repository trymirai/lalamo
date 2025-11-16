from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree

from .common import StateLayerBase

__all__ = ["Mamba2StateLayer"]


class Mamba2StateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch tokens conv_channels"]
    ssm_state: Float[Array, "*batch groups head_channels state_channels"]

    def __post_init__(self) -> None:
        if self.conv_state.ndim not in (2, 3):
            raise ValueError(
                f"Conv state must have 2 or 3 dimensions: [batch], tokens, conv_channels,"
                f" got shape {self.conv_state.shape}",
            )
        if self.ssm_state.ndim not in (3, 4):
            raise ValueError(
                f"SSM state must have 3 or 4 dimensions: [batch], groups, head_channels, state_channels,"
                f" got shape {self.ssm_state.shape}",
            )
        if self.conv_state.dtype != self.ssm_state.dtype:
            raise ValueError("Conv state and SSM state must have the same dtype")

    @classmethod
    def init(
        cls,
        kernel_size: int,
        inner_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        state_dim: int,
        dtype: DTypeLike,
    ) -> Self:
        return cls(
            conv_state=jnp.zeros((kernel_size - 1, inner_dim + 2 * num_groups * state_dim), dtype=dtype),
            ssm_state=jnp.zeros((num_heads, head_dim, state_dim), dtype=dtype),
        )

    def export(self) -> ParameterTree:
        return dict(
            conv_state=self.conv_state,
            ssm_state=self.ssm_state,
        )
