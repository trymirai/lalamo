import math
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.utils.sharding import use_out_sharding

__all__ = [
    "CausalConvResult",
    "SeparableCausalConv",
    "SeparableCausalConvConfig",
]


class CausalConvResult(NamedTuple):
    outputs: Float[Array, "*batch suffix_tokens channels"]
    state: Float[Array, "*batch tokens channels"] | None = None


@dataclass(frozen=True)
class SeparableCausalConvConfig(LalamoConfig):
    has_biases: bool

    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        kernel_size: int,
    ) -> "SeparableCausalConv":
        scale = 1 / math.sqrt(kernel_size * input_dim)
        weights = initializer.normal(scale, (input_dim, kernel_size))
        if self.has_biases:
            biases = initializer.zeros((input_dim,))
        else:
            biases = None
        return SeparableCausalConv(
            config=self,
            weights=weights,
            biases=biases,
        )


class SeparableCausalConv(LalamoModule[SeparableCausalConvConfig]):
    weights: Float[Array, "channels kernel"]
    biases: Float[Array, " channels"] | None

    def __post_init__(self) -> None:
        input_dim, _ = self.weights.shape
        if self.biases is not None:
            (output_dim,) = self.biases.shape
            if output_dim != input_dim:
                raise ValueError(
                    f"Output dimension of biases ({output_dim}) must match input dimension ({input_dim})",
                )

    @property
    def input_dim(self) -> int:
        input_dim, _ = self.weights.shape
        return input_dim

    @property
    def kernel_size(self) -> int:
        _, kernel_size = self.weights.shape
        return kernel_size

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        length_without_padding: Int[Array, ""] | int | None = None,
        state: Float[Array, "prefix_tokens channels"] | None = None,
        return_updated_state: bool = False,
    ) -> CausalConvResult:
        num_suffix_tokens, input_dim = inputs.shape

        if state is None:
            state = jnp.zeros_like(jnp.broadcast_to(inputs[:1], (self.kernel_size - 1, input_dim)))

        required_context = num_suffix_tokens + self.kernel_size - 1

        inputs_with_history = _causal_conv_context(state, inputs)
        conv_outputs = _separable_causal_conv(
            inputs_with_history[None, -required_context:, :],
            self.weights.astype(inputs.dtype),
        )

        results = conv_outputs.squeeze(0)
        if self.biases is not None:
            results = _add_conv_biases(results, self.biases.astype(results.dtype))

        if return_updated_state:
            if length_without_padding is None:
                length_without_padding = num_suffix_tokens
            length_without_padding = jnp.asarray(length_without_padding, dtype=jnp.int32)
            length_without_padding = jnp.clip(length_without_padding, 0, num_suffix_tokens)
            updated_state = _updated_causal_conv_state(inputs_with_history, inputs, length_without_padding)
        else:
            updated_state = None

        return CausalConvResult(
            results,
            updated_state,
        )

    def step(
        self,
        token: Float[Array, " channels"],
        state: Float[Array, "kernel_minus_1 channels"],
    ) -> tuple[Float[Array, " channels"], Float[Array, "kernel_minus_1 channels"]]:
        """Single-token conv update without full convolution overhead."""
        full_input = jnp.concatenate([state, token[None, :]], axis=0)
        output = einsum(full_input, self.weights.astype(token.dtype), "kernel channels, channels kernel -> channels")
        if self.biases is not None:
            output = output + self.biases.astype(output.dtype)
        new_state = jnp.concatenate([state[1:], token[None, :]], axis=0)
        return output, new_state


def _separable_causal_conv(
    inputs: Float[Array, "batch context_tokens channels"],
    weights: Float[Array, "channels kernel"],
) -> Float[Array, "batch suffix_tokens channels"]:
    input_dim, _ = weights.shape
    return jax.lax.conv_general_dilated(
        inputs,
        weights[:, :, None],
        window_strides=(1,),
        feature_group_count=input_dim,
        padding="VALID",
        dimension_numbers=("NTC", "OTI", "NTC"),
    )


@use_out_sharding((None, None))
def _causal_conv_context(
    state: Float[Array, "state_tokens channels"],
    inputs: Float[Array, "suffix_tokens channels"],
) -> Float[Array, "context_tokens channels"]:
    return jnp.concatenate([state, inputs], axis=0)


@use_out_sharding((None, None))
def _add_conv_biases(
    outputs: Float[Array, "suffix_tokens channels"],
    biases: Float[Array, " channels"],
) -> Float[Array, "suffix_tokens channels"]:
    return outputs + biases


@use_out_sharding((None, None))
def _updated_causal_conv_state(
    inputs_with_history: Float[Array, "context_tokens channels"],
    inputs: Float[Array, "suffix_tokens channels"],
    length_without_padding: Int[Array, ""] | int,
) -> Float[Array, "state_tokens channels"]:
    return jax.lax.dynamic_slice_in_dim(
        inputs_with_history,
        start_index=length_without_padding,
        slice_size=inputs_with_history.shape[0] - inputs.shape[0],
        axis=0,
    )
