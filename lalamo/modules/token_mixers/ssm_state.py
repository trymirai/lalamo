from typing import Self

import jax
import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, Float, Int

from lalamo.modules.token_mixer import StateLayerBase

__all__ = ["LaggedSSMStateLayer", "SSMStateLayer", "fold_lag_factors"]


def fold_lag_factors(
    conv_state: Float[Array, "history conv_channels"],
    ssm_state: Float[Array, "heads value_channels key_channels"],
    keys: Float[Array, "nodes heads key_channels"],
    update_values: Float[Array, "nodes heads value_channels"],
    prop_updates: Float[Array, "nodes heads key_channels"],
    cumulative_decay: Float[Array, "nodes heads"],
    conv_windows: Float[Array, "nodes history conv_channels"],
    accepted_node_indices: Int[Array, " nodes"],
    num_accepted_nodes: Int[Array, ""],
) -> tuple[
    Float[Array, "history conv_channels"],
    Float[Array, "heads value_channels key_channels"],
]:
    num_nodes, _ = cumulative_decay.shape
    *_, key_channels = keys.shape
    node_indices = jnp.arange(num_nodes, dtype=accepted_node_indices.dtype)
    on_path = jnp.any(accepted_node_indices[None, :] == node_indices[:, None], axis=1)
    leaf = jnp.maximum(accepted_node_indices[jnp.maximum(num_accepted_nodes - 1, 0)], 0)

    leaf_decay = cumulative_decay[leaf]
    relative_decay = leaf_decay[None, :] - cumulative_decay
    path_weights = jnp.where(on_path[:, None], jnp.exp(jnp.where(on_path[:, None], relative_decay, 0.0)), 0.0)

    path_prop = jnp.eye(key_channels, dtype=jnp.float32)[None, :, :] - einsum(
        jnp.where(on_path[:, None, None], prop_updates, 0.0),
        keys,
        "nodes heads key_channels_in, nodes heads key_channels_out -> heads key_channels_in key_channels_out",
    )
    folded_ssm_state = jnp.exp(leaf_decay)[:, None, None] * einsum(
        ssm_state.astype(jnp.float32),
        path_prop,
        "heads value_channels key_channels_in, heads key_channels_in key_channels_out"
        " -> heads value_channels key_channels_out",
    ) + einsum(
        path_weights,
        update_values,
        keys,
        "nodes heads, nodes heads value_channels, nodes heads key_channels -> heads value_channels key_channels",
    )

    has_accepted_nodes = num_accepted_nodes > 0
    new_ssm_state = jnp.where(has_accepted_nodes, folded_ssm_state.astype(ssm_state.dtype), ssm_state)
    new_conv_state = jnp.where(has_accepted_nodes, conv_windows[leaf], conv_state)
    return new_conv_state, new_ssm_state


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
    ) -> Self:
        conv_state = jnp.zeros((kernel_size - 1, conv_dim), dtype=jnp.float32)
        ssm_state = jnp.zeros(ssm_state_shape, dtype=jnp.float32)
        return cls(conv_state=conv_state, ssm_state=ssm_state)

    def begin_verification(self, num_nodes: int) -> "LaggedSSMStateLayer":
        return LaggedSSMStateLayer.wrap(self, num_nodes)


class LaggedSSMStateLayer(SSMStateLayer):
    keys: Float[Array, "*batch nodes heads key_channels"]
    update_values: Float[Array, "*batch nodes heads value_channels"]
    prop_updates: Float[Array, "*batch nodes heads key_channels"]
    cumulative_decay: Float[Array, "*batch nodes heads"]
    conv_windows: Float[Array, "*batch nodes tokens conv_channels"]

    @classmethod
    def wrap(cls, committed: SSMStateLayer, num_nodes: int) -> "LaggedSSMStateLayer":
        *batch, history, conv_channels = committed.conv_state.shape
        *_, num_heads, value_channels, key_channels = committed.ssm_state.shape
        return cls(
            conv_state=committed.conv_state,
            ssm_state=committed.ssm_state,
            keys=jnp.zeros((*batch, num_nodes, num_heads, key_channels), dtype=jnp.float32),
            update_values=jnp.zeros((*batch, num_nodes, num_heads, value_channels), dtype=jnp.float32),
            prop_updates=jnp.zeros((*batch, num_nodes, num_heads, key_channels), dtype=jnp.float32),
            cumulative_decay=jnp.zeros((*batch, num_nodes, num_heads), dtype=jnp.float32),
            conv_windows=jnp.zeros(
                (*batch, num_nodes, history, conv_channels),
                dtype=committed.conv_state.dtype,
            ),
        )

    def begin_verification(self, num_nodes: int) -> "LaggedSSMStateLayer":
        del num_nodes
        return self

    def commit_accepted(
        self,
        accepted_node_indices: Int[Array, "batch nodes"],
        num_accepted_nodes: Int[Array, " batch"],
    ) -> SSMStateLayer:
        fold = fold_lag_factors if self.conv_state.ndim == 2 else jax.vmap(fold_lag_factors)
        conv_state, ssm_state = fold(
            self.conv_state,
            self.ssm_state,
            self.keys,
            self.update_values,
            self.prop_updates,
            self.cumulative_decay,
            self.conv_windows,
            accepted_node_indices,
            num_accepted_nodes,
        )
        return SSMStateLayer(conv_state=conv_state, ssm_state=ssm_state)
