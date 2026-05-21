import math
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.utils.sharding import sharding_of

__all__ = [
    "CausalConvResult",
    "ConvPrecision",
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
        dtype: DTypeLike,
    ) -> "SeparableCausalConv":
        scale = 1 / math.sqrt(kernel_size * input_dim)
        weights = initializer.normal(scale, (input_dim, kernel_size), dtype=dtype)
        if self.has_biases:
            biases = initializer.zeros((input_dim,), dtype=dtype)
        else:
            biases = None
        return SeparableCausalConv(
            config=self,
            sharding_config=initializer.sharding_config,
            weights=weights,
            biases=biases,
        )


class ConvPrecision(StrEnum):
    MATCH_WEIGHTS = "match_weights"
    MATCH_INPUTS = "match_input"


class SeparableCausalConv(LalamoModule[SeparableCausalConvConfig]):
    weights: Float[Array, "channels kernel"]
    biases: Float[Array, " channels"] | None

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
        precision: ConvPrecision = ConvPrecision.MATCH_INPUTS,
    ) -> CausalConvResult:
        match precision:
            case ConvPrecision.MATCH_WEIGHTS:
                dtype = self.weights.dtype
            case ConvPrecision.MATCH_INPUTS:
                dtype = inputs.dtype

        inputs = inputs.astype(dtype)

        num_suffix_tokens, input_dim = inputs.shape

        if state is None:
            state = jnp.zeros_like(jnp.broadcast_to(inputs[:1], (self.kernel_size - 1, input_dim)))

        required_context = num_suffix_tokens + self.kernel_size - 1

        inputs_with_history = _causal_conv_context(state, inputs)
        conv_outputs = _separable_causal_conv(
            inputs_with_history[None, -required_context:, :],
            self.weights.astype(dtype),
        )

        results = conv_outputs.squeeze(0)
        if self.biases is not None:
            results = results + self.biases.astype(dtype)

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
        precision: ConvPrecision = ConvPrecision.MATCH_INPUTS,
    ) -> tuple[Float[Array, " channels"], Float[Array, "kernel_minus_1 channels"]]:
        match precision:
            case ConvPrecision.MATCH_WEIGHTS:
                dtype = self.weights.dtype
            case ConvPrecision.MATCH_INPUTS:
                dtype = token.dtype

        assert state.dtype == dtype

        full_input = jnp.concatenate([state, token[None, :].astype(dtype)], axis=0)
        output = einsum(full_input, self.weights.astype(dtype), "kernel channels, channels kernel -> channels")
        if self.biases is not None:
            output = output + self.biases.astype(dtype)
        new_state = jnp.concatenate([state[1:], token[None, :]], axis=0)
        return output, new_state


def _separable_causal_conv_impl(
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
        out_sharding=sharding_of(inputs),
    )


@jax.custom_vjp
def _separable_causal_conv(
    inputs: Float[Array, "batch context_tokens channels"],
    weights: Float[Array, "channels kernel"],
) -> Float[Array, "batch suffix_tokens channels"]:
    return _separable_causal_conv_impl(inputs, weights)


def _causal_conv_windows(
    inputs: Float[Array, "*batch context_tokens channels"],
    kernel_size: int,
) -> Float[Array, "*batch suffix_tokens channels kernel"]:
    context_tokens = inputs.shape[-2]
    suffix_tokens = context_tokens - kernel_size + 1
    suffix_positions = jnp.arange(suffix_tokens)
    kernel_positions = jnp.arange(kernel_size)
    token_indices = suffix_positions[:, None] + kernel_positions[None, :]
    windows = jnp.take(inputs, token_indices, axis=-2)
    return rearrange(windows, "... suffix_tokens kernel channels -> ... suffix_tokens channels kernel")


def _separable_causal_conv_forward(
    inputs: Float[Array, "batch context_tokens channels"],
    weights: Float[Array, "channels kernel"],
) -> tuple[
    Float[Array, "batch suffix_tokens channels"],
    tuple[Float[Array, "batch context_tokens channels"], Float[Array, "channels kernel"]],
]:
    outputs = _separable_causal_conv_impl(inputs, weights)
    return outputs, (inputs, weights)


def _separable_causal_conv_backward(
    residuals: tuple[Float[Array, "*sample batch context_tokens channels"], Float[Array, "channels kernel"]],
    output_gradients: Float[Array, "*sample batch suffix_tokens channels"],
) -> tuple[Float[Array, "*sample batch context_tokens channels"], Float[Array, "*sample channels kernel"]]:
    inputs, weights = residuals
    _, kernel_size = weights.shape
    input_windows = _causal_conv_windows(inputs, kernel_size)
    weight_gradients = einsum(
        output_gradients,
        input_windows,
        "... batch suffix_tokens channels, ... batch suffix_tokens channels kernel -> ... channels kernel",
    )
    output_gradient_padding = ((0, 0),) * (output_gradients.ndim - 2) + (
        (kernel_size - 1, kernel_size - 1),
        (0, 0),
    )
    padded_output_gradients = jnp.pad(output_gradients, output_gradient_padding)
    output_gradient_windows = _causal_conv_windows(padded_output_gradients, kernel_size)
    input_gradients = einsum(
        output_gradient_windows,
        jnp.flip(weights, axis=-1),
        "... context_tokens channels kernel, channels kernel -> ... context_tokens channels",
    )
    return input_gradients, weight_gradients


_separable_causal_conv.defvjp(_separable_causal_conv_forward, _separable_causal_conv_backward)


def _causal_conv_context(
    state: Float[Array, "state_tokens channels"],
    inputs: Float[Array, "suffix_tokens channels"],
) -> Float[Array, "context_tokens channels"]:
    return jnp.concatenate([state, inputs], axis=0)


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
