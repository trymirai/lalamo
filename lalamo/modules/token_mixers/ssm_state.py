from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.modules.token_mixer import StateLayerBase
from lalamo.utils.sharding import LogicalAxis, ShardingConfig

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
        batch_size: int,
        kernel_size: int,
        conv_dim: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        sharding_config: ShardingConfig,
    ) -> Self:
        conv_sharding = sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None))
        ssm_sharding = sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None, None))
        conv_state = jnp.zeros((batch_size, kernel_size - 1, conv_dim), dtype=jnp.float32, out_sharding=conv_sharding)
        ssm_state = jnp.zeros(
            (batch_size, num_heads, key_dim, value_dim), dtype=jnp.float32, out_sharding=ssm_sharding
        )
        return cls(conv_state=conv_state, ssm_state=ssm_state)
