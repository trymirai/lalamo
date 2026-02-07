from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree

from .common import StateLayerBase

__all__ = ["SSMStateLayer"]


class SSMStateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch tokens conv_channels"]
    ssm_state: Float[Array, "*batch groups head_channels state_channels"]

    def __post_init__(self) -> None:
        if self.conv_state.ndim not in (2, 3):
            raise ValueError(
                "Conv state must have 2 or 3 dimensions: [batch], tokens, conv_channels,"
                f" got shape {self.conv_state.shape}",
            )
        if self.ssm_state.ndim not in (3, 4):
            raise ValueError(
                "SSM state must have 3 or 4 dimensions: [batch], heads, head_channels, state_channels,"
                f" got shape {self.ssm_state.shape}",
            )
        if self.conv_state.dtype != self.ssm_state.dtype:
            raise ValueError("Conv state and SSM state must have the same dtype")

    @classmethod
    def init(
        cls,
        kernel_size: int,
        conv_dim: int,
        ssm_state_shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> Self:
        conv_state = jnp.zeros((kernel_size - 1, conv_dim), dtype=dtype)
        ssm_state = jnp.zeros(ssm_state_shape, dtype=dtype)
        return cls(conv_state=conv_state, ssm_state=ssm_state)

    def export(self) -> ParameterTree:
        return dict(
            conv_state=self.conv_state,
            ssm_state=self.ssm_state,
        )
