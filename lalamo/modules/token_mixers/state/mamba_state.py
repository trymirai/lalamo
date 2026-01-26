from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from .common import StateLayerBase

__all__ = ["Mamba2StateLayer"]


class Mamba2StateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch tokens conv_channels"]
    ssm_state: Float[Array, "*batch groups head_channels state_channels"]

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
