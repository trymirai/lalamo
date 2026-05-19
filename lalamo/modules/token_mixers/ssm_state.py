from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.modules.token_mixer import StateLayerBase

__all__ = ["SSMStateLayer"]


class SSMStateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch tokens conv_channels"]
    ssm_state: Float[Array, "*batch heads value_channels key_channels"]

    def __post_init__(self) -> None:
        if self.conv_state.ndim not in (2, 3):
            raise ValueError(
                "Conv state must have 2 or 3 dimensions: [batch], tokens, conv_channels,"
                f" got shape {self.conv_state.shape}",
            )
        if self.ssm_state.ndim not in (3, 4):
            raise ValueError(
                "SSM state must have 3 or 4 dimensions: [batch], heads, state_channels, head_channels,"
                f" got shape {self.ssm_state.shape}",
            )
    @classmethod
    def init(
        cls,
        kernel_size: int,
        conv_dim: int,
        ssm_state_shape: tuple[int, ...],
        dtype: DTypeLike,
        ssm_dtype: DTypeLike | None = None,
    ) -> Self:
        conv_state = jnp.zeros((kernel_size - 1, conv_dim), dtype=dtype)
        ssm_state = jnp.zeros(ssm_state_shape, dtype=ssm_dtype or dtype)
        return cls(conv_state=conv_state, ssm_state=ssm_state)
