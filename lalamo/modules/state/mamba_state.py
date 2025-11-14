from typing import Self

import jax.numpy as jnp
from jax.lax import dynamic_slice_in_dim
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.common import ParameterTree

from .common import StateLayerBase

__all__ = ["MambaStateLayer"]


class MambaStateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch history conv_channels"]
    ssm_state: Float[Array, "*batch heads head_channels state_channels"]

    def __post_init__(self) -> None:
        if self.conv_state.ndim not in (2, 3):
            raise ValueError(
                f"Conv state must have 2 or 3 dimensions: [batch], history, conv_channels,"
                f" got shape {self.conv_state.shape}",
            )
        if self.ssm_state.ndim not in (3, 4):
            raise ValueError(
                f"SSM state must have 3 or 4 dimensions: [batch], heads, head_channels, state_channels,"
                f" got shape {self.ssm_state.shape}",
            )
        if self.conv_state.dtype != self.ssm_state.dtype:
            raise ValueError("Conv state and SSM state must have the same dtype")

    def _raise_if_batched(self) -> None:
        if self.conv_state.ndim != 2:
            raise ValueError(
                "Attempted to call a method on a batched version of MambaStateLayer. Use vmap instead.",
            )

    @classmethod
    def init(
        cls,
        conv_state: Float[Array, "history conv_channels"],
        ssm_state: Float[Array, "heads head_channels state_channels"],
    ) -> "MambaStateLayer":
        return cls(conv_state, ssm_state)

    def extend(
        self,
        conv_input: Float[Array, "suffix_tokens conv_channels"],
        ssm_state_update: Float[Array, "heads head_channels state_channels"],
        added_length: Int[Array, ""] | int | None = None,
    ) -> "MambaStateLayer":
        self._raise_if_batched()
        conv_history_length, conv_channels = self.conv_state.shape
        num_new_tokens, _ = conv_input.shape
        if added_length is None:
            added_length = num_new_tokens

        combined = jnp.concatenate([self.conv_state, conv_input], axis=0)
        start = jnp.asarray(added_length, dtype=jnp.int32)
        updated_conv_state = dynamic_slice_in_dim(combined, start, conv_history_length, axis=0)

        return MambaStateLayer(updated_conv_state, ssm_state_update)

    @classmethod
    def empty(
        cls,
        conv_history_length: int,
        conv_channels: int,
        num_heads: int,
        head_dim: int,
        state_dim: int,
        dtype: DTypeLike,
    ) -> Self:
        return cls(
            conv_state=jnp.zeros((conv_history_length, conv_channels), dtype=dtype),
            ssm_state=jnp.zeros((num_heads, head_dim, state_dim), dtype=dtype),
        )

    def export(self) -> ParameterTree:
        return dict(
            conv_state=self.conv_state,
            ssm_state=self.ssm_state,
        )
