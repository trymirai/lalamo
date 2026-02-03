import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_array
from lalamo.modules.common import LalamoModule

__all__ = [
    "CausalConvResult",
    "SeparableCausalConv",
    "SeparableCausalConvConfig",
]


class CausalConvResult(NamedTuple):
    outputs: Float[Array, "*batch suffix_tokens channels"]
    state: Float[Array, "*batch tokens channels"] | None = None


@dataclass(frozen=True)
class SeparableCausalConvConfig:
    precision: DTypeLike
    has_biases: bool

    def random_init(
        self,
        input_dim: int,
        kernel_size: int,
        *,
        key: PRNGKeyArray,
    ) -> "SeparableCausalConv":
        scale = 1 / math.sqrt(kernel_size * input_dim)
        weights = jax.random.uniform(
            key,
            (input_dim, kernel_size),
            minval=-scale,
            maxval=scale,
            dtype=self.precision,
        )
        if self.has_biases:
            biases = jnp.zeros((input_dim,), dtype=self.precision)
        else:
            biases = None
        return SeparableCausalConv(self, weights=weights, biases=biases)

    def empty(
        self,
        input_dim: int,
        kernel_size: int,
    ) -> "SeparableCausalConv":
        weights = dummy_array(
            (input_dim, kernel_size),
            dtype=self.precision,
        )
        if self.has_biases:
            biases = dummy_array((input_dim,), dtype=self.precision)
        else:
            biases = None
        return SeparableCausalConv(self, weights=weights, biases=biases)


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
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

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

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        length_without_padding: Int[Array, ""] | int | None = None,
        state: Float[Array, "prefix_tokens channels"] | None = None,
        return_updated_state: bool = False,
    ) -> CausalConvResult:
        num_suffix_tokens, input_dim = inputs.shape

        if state is None:
            state = jnp.zeros((self.kernel_size - 1, input_dim), dtype=inputs.dtype)

        required_context = num_suffix_tokens + self.kernel_size - 1

        inputs_with_history = jnp.concatenate([state, inputs], axis=0)
        conv_outputs = jax.lax.conv_general_dilated(
            inputs_with_history[None, -required_context:, :],
            self.weights[:, :, None],
            window_strides=(1,),
            feature_group_count=input_dim,
            padding="VALID",
            dimension_numbers=("NTC", "OTI", "NTC"),
        )

        results = conv_outputs.squeeze(0)
        if self.biases is not None:
            results = results + self.biases

        if return_updated_state:
            if length_without_padding is None:
                length_without_padding = num_suffix_tokens
            length_without_padding = jnp.asarray(length_without_padding, dtype=jnp.int32)
            length_without_padding = jnp.clip(length_without_padding, 0, num_suffix_tokens)
            updated_state = jax.lax.dynamic_slice_in_dim(
                inputs_with_history,
                start_index=length_without_padding,
                slice_size=self.kernel_size - 1,
                axis=0,
            )
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
        output = einsum(full_input, self.weights, "kernel channels, channels kernel -> channels")
        if self.biases is not None:
            output = output + self.biases
        new_state = jnp.concatenate([state[1:], token[None, :]], axis=0)
        return output, new_state

    def export_weights(self) -> ParameterTree:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> "SeparableCausalConv":
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=require_array(weights["biases"]) if self.biases is not None else None,
        )
