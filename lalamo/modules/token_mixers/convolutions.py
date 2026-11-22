import math
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import einsum, rearrange, repeat
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule

__all__ = [
    "CausalConv",
    "CausalConvConfig",
    "CausalConvResult",
    "ConvPrecision",
    "SeparableCausalConv",
    "SeparableCausalConvConfig",
]


class CausalConvResult(NamedTuple):
    outputs: Float[Array, "*batch suffix_tokens channels"]
    state: Float[Array, "*batch tokens channels"] | None = None


@dataclass(frozen=True)
class CausalConvConfig(LalamoConfig):
    """Stateful depthwise causal convolution computed in float32.

    `weights[:, kernel_size - 1]` weights the current token. With `num_dynamic_groups`, every token adds its own
    taps, shared by contiguous channel groups. The identity adds the input before the output scale.
    """

    kernel_size: int
    num_dynamic_groups: int | None
    has_identity: bool
    output_scale: float | None

    def init(self, initializer: Initializer, input_dim: int) -> "CausalConv":
        if self.num_dynamic_groups is not None and input_dim % self.num_dynamic_groups != 0:
            raise ValueError(f"{self.num_dynamic_groups} dynamic groups must divide {input_dim} channels.")
        scale = 1 / math.sqrt(self.kernel_size)
        weights = initializer.normal(scale, (input_dim, self.kernel_size), dtype=jnp.float32)
        return CausalConv(config=self, sharding_config=initializer.sharding_config, weights=weights)


class CausalConv(LalamoModule[CausalConvConfig]):
    weights: Float[Array, "channels kernel"]

    @property
    def input_dim(self) -> int:
        input_dim, _ = self.weights.shape
        return input_dim

    @property
    def kernel_size(self) -> int:
        _, kernel_size = self.weights.shape
        return kernel_size

    def init_static_state(self) -> Float[Array, "kernel_minus_one channels"]:
        return jnp.zeros((self.kernel_size - 1, self.input_dim), dtype=jnp.float32)

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        state: Float[Array, "kernel_minus_one channels"] | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        *,
        coefficient_deltas: Float[Array, "suffix_tokens kernel groups"] | None = None,
    ) -> CausalConvResult:
        if (coefficient_deltas is None) != (self.config.num_dynamic_groups is None):
            raise ValueError("Coefficient deltas must be passed exactly when the convolution has dynamic groups.")
        num_suffix_tokens, input_dim = inputs.shape
        upcasted_inputs = inputs.astype(jnp.float32)
        if state is None:
            state = self.init_static_state()
        inputs_with_history = _causal_conv_context(state.astype(jnp.float32), upcasted_inputs)
        windows = _causal_conv_windows(inputs_with_history, self.kernel_size)
        weights = self.weights.astype(jnp.float32)[None]
        if coefficient_deltas is not None:
            weights = weights + repeat(
                coefficient_deltas.astype(jnp.float32),
                "suffix_tokens kernel groups -> suffix_tokens (groups group_size) kernel",
                group_size=input_dim // coefficient_deltas.shape[-1],
            )
        outputs = jnp.sum(windows * weights, axis=-1)
        if self.config.has_identity:
            outputs = outputs + upcasted_inputs
        if self.config.output_scale is not None:
            outputs = outputs * self.config.output_scale

        if return_updated_state:
            if length_without_padding is None:
                length_without_padding = num_suffix_tokens
            length_without_padding = jnp.clip(
                jnp.asarray(length_without_padding, dtype=jnp.int32), 0, num_suffix_tokens
            )
            updated_state = _updated_causal_conv_state(inputs_with_history, upcasted_inputs, length_without_padding)
        else:
            updated_state = None
        return CausalConvResult(outputs.astype(inputs.dtype), updated_state)


@dataclass(frozen=True)
class SeparableCausalConvConfig(LalamoConfig):
    has_biases: bool

    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        kernel_size: int,
        *,
        dtype: DTypeLike | None = jnp.float32,
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
        *,
        coefficient_deltas: Float[Array, "suffix_tokens kernel groups"] | None = None,
    ) -> CausalConvResult:
        match precision:
            case ConvPrecision.MATCH_WEIGHTS:
                output_dtype = self.weights.dtype
            case ConvPrecision.MATCH_INPUTS:
                output_dtype = inputs.dtype

        state_dtype = inputs.dtype if state is None else state.dtype
        inputs_for_state = inputs.astype(state_dtype)

        num_suffix_tokens, input_dim = inputs_for_state.shape

        if state is None:
            state = jnp.zeros_like(jnp.broadcast_to(inputs_for_state[:1], (self.kernel_size - 1, input_dim)))
        else:
            state = state.astype(state_dtype)

        required_context = num_suffix_tokens + self.kernel_size - 1

        inputs_with_history = _causal_conv_context(state, inputs_for_state)
        weights = self.weights.astype(output_dtype)[None]
        if coefficient_deltas is not None:
            weights = weights + repeat(
                coefficient_deltas.astype(output_dtype),
                "suffix_tokens kernel groups -> suffix_tokens (groups group_size) kernel",
                suffix_tokens=num_suffix_tokens,
                kernel=self.kernel_size,
                group_size=self.input_dim // coefficient_deltas.shape[-1],
            )
        input_windows = _causal_conv_windows(
            inputs_with_history[-required_context:].astype(output_dtype), self.kernel_size
        )
        accumulation_dtype = np.promote_types(output_dtype, np.float32)
        results = jnp.sum(
            input_windows.astype(accumulation_dtype) * weights.astype(accumulation_dtype), axis=-1
        ).astype(output_dtype)
        if self.biases is not None:
            results = results + self.biases.astype(output_dtype)

        if return_updated_state:
            if length_without_padding is None:
                length_without_padding = num_suffix_tokens
            length_without_padding = jnp.asarray(length_without_padding, dtype=jnp.int32)
            length_without_padding = jnp.clip(length_without_padding, 0, num_suffix_tokens)
            updated_state = _updated_causal_conv_state(
                inputs_with_history,
                inputs_for_state,
                length_without_padding,
            ).astype(state_dtype)
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

        token = token.astype(dtype)
        full_input = jnp.concatenate([state, token[None, :]], axis=0)
        output = einsum(full_input, self.weights.astype(dtype), "kernel channels, channels kernel -> channels")
        if self.biases is not None:
            output = output + self.biases.astype(dtype)
        new_state = jnp.concatenate([state[1:], token[None, :]], axis=0)
        return output, new_state


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
