from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array
from lalamo.modules.activations import Activation
from lalamo.modules.common import PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.state import MambaStateLayer

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult

__all__ = [
    "Mamba2",
    "Mamba2Config",
    "Mamba2Result",
]


Mamba2Result = TokenMixerResult[MambaStateLayer]


class CausalConv1d(eqx.Module):
    weight: Float[Array, "total_channels kernel"]
    bias: Float[Array, " total_channels"] | None

    output_dims: tuple[int, ...] = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens total_channels"],
        state: Float[Array, "history total_channels"] | None = None,
    ) -> tuple[tuple[Float[Array, "suffix_tokens channels"], ...], Float[Array, "history total_channels"]]:
        num_tokens, num_channels = inputs.shape

        if state is None:
            state = jnp.zeros((self.kernel_size, num_channels), dtype=inputs.dtype)

        padded = jnp.concatenate([state, inputs], axis=0)

        def apply_kernel(idx: Int[Array, ""]) -> Float[Array, " total_channels"]:
            window = jax.lax.dynamic_slice(padded, (idx, 0), (self.kernel_size, num_channels))
            result = einsum(window, self.weight, "kernel channels, channels kernel -> channels")
            if self.bias is not None:
                result = result + self.bias
            return result

        outputs = vmap(apply_kernel)(jnp.arange(num_tokens))
        updated_state = jax.lax.dynamic_slice(padded, (num_tokens, 0), (self.kernel_size, num_channels))

        split_outputs = tuple(jnp.split(outputs, self._get_split_points(self.output_dims), axis=-1))
        return split_outputs, updated_state

    @staticmethod
    def _get_split_points(output_dims: Sequence[int]) -> tuple[int, ...]:
        result = []
        last_split_point = 0
        for dim in output_dims[:-1]:
            last_split_point += dim
            result.append(last_split_point)
        return tuple(result)


@dataclass(frozen=True)
class Mamba2Config(TokenMixerConfigBase):
    in_projection_config: LinearConfig
    out_projection_config: LinearConfig

    num_value_heads: int
    num_query_key_heads: int
    head_dim: int
    state_dim: int
    conv_kernel_size: int
    expand: int
    chunk_size: int

    activation: Activation
    has_in_biases: bool
    has_out_biases: bool
    has_conv_biases: bool

    @property
    def inner_dim(self) -> int:
        return self.num_value_heads * self.head_dim

    @property
    def rope_dim(self) -> int:
        return 0

    def random_init(
        self,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "Mamba2":
        in_key, out_key, conv_key, skip_key = jax.random.split(key, 4)

        in_projection = self.in_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(
                self.inner_dim + 2 * self.num_query_key_heads * self.state_dim,
                self.inner_dim,
                self.num_value_heads,
            ),
            has_biases=self.has_in_biases,
            key=in_key,
        )

        out_projection = self.out_projection_config.random_init(
            self.inner_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
            key=out_key,
        )

        conv_channels = self.inner_dim + 2 * self.num_query_key_heads * self.state_dim
        conv_weight = jax.random.uniform(
            conv_key,
            (conv_channels, self.conv_kernel_size),
            minval=-1.0,
            maxval=1.0,
            dtype=in_projection.activation_precision,
        )

        if self.has_conv_biases:
            conv_bias = jnp.zeros((conv_channels,), dtype=in_projection.activation_precision)
        else:
            conv_bias = None

        conv = CausalConv1d(
            weight=conv_weight,
            bias=conv_bias,
            output_dims=(
                self.inner_dim,
                self.num_query_key_heads * self.state_dim,
                self.num_query_key_heads * self.state_dim,
            ),
            kernel_size=self.conv_kernel_size,
        )

        skip_connection_weight = jax.random.normal(
            skip_key,
            (self.num_value_heads,),
            dtype=in_projection.activation_precision,
        )

        gate_bias = jnp.zeros((self.inner_dim,), dtype=in_projection.activation_precision)

        return Mamba2(
            self,
            in_projection=in_projection,
            conv=conv,
            out_projection=out_projection,
            skip_connection_weight=skip_connection_weight,
            gate_bias=gate_bias,
            num_value_heads=self.num_value_heads,
            num_query_key_heads=self.num_query_key_heads,
            head_dim=self.head_dim,
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
        )

    def empty(
        self,
        model_dim: int,
    ) -> "Mamba2":
        in_projection = self.in_projection_config.empty(
            input_dim=model_dim,
            output_dims=(
                self.inner_dim + 2 * self.num_query_key_heads * self.state_dim,
                self.inner_dim,
                self.num_value_heads,
            ),
            has_biases=self.has_in_biases,
        )

        out_projection = self.out_projection_config.empty(
            self.inner_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
        )

        conv_channels = self.inner_dim + 2 * self.num_query_key_heads * self.state_dim
        conv_weight = dummy_array((conv_channels, self.conv_kernel_size), in_projection.activation_precision)

        if self.has_conv_biases:
            conv_bias = dummy_array((conv_channels,), in_projection.activation_precision)
        else:
            conv_bias = None

        conv = CausalConv1d(
            weight=conv_weight,
            bias=conv_bias,
            output_dims=(
                self.inner_dim,
                self.num_query_key_heads * self.state_dim,
                self.num_query_key_heads * self.state_dim,
            ),
            kernel_size=self.conv_kernel_size,
        )

        skip_connection_weight = dummy_array((self.num_value_heads,), in_projection.activation_precision)
        gate_bias = dummy_array((self.inner_dim,), in_projection.activation_precision)

        return Mamba2(
            self,
            in_projection=in_projection,
            conv=conv,
            out_projection=out_projection,
            skip_connection_weight=skip_connection_weight,
            gate_bias=gate_bias,
            num_value_heads=self.num_value_heads,
            num_query_key_heads=self.num_query_key_heads,
            head_dim=self.head_dim,
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
        )


class Mamba2(TokenMixerBase[Mamba2Config, MambaStateLayer]):
    in_projection: LinearBase
    conv: CausalConv1d
    out_projection: LinearBase

    skip_connection_weight: Float[Array, " value_heads"]
    gate_bias: Float[Array, " inner_dim"]

    num_value_heads: int = eqx.field(static=True)
    num_query_key_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.in_projection.activation_precision

    @property
    def model_dim(self) -> int:
        return self.in_projection.input_dim

    @property
    def inner_dim(self) -> int:
        return self.num_value_heads * self.head_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return PositionalEmbeddingSelector.NONE

    def __post_init__(self) -> None:
        if self.skip_connection_weight.shape != (self.num_value_heads,):
            raise ValueError(
                f"Skip connection weight must have shape (num_value_heads,) = ({self.num_value_heads},), "
                f"got {self.skip_connection_weight.shape}",
            )
        if self.gate_bias.shape != (self.inner_dim,):
            raise ValueError(
                f"Gate bias must have shape (inner_dim,) = ({self.inner_dim},), got {self.gate_bias.shape}",
            )
        if self.num_value_heads % self.num_query_key_heads != 0:
            raise ValueError(
                f"Number of value heads ({self.num_value_heads}) must be divisible by "
                f"number of query-key heads ({self.num_query_key_heads})",
            )

    def _chunk_scan(
        self,
        hidden_states: Float[Array, "suffix_tokens value_heads head_channels"],
        input_projection: Float[Array, "suffix_tokens query_key_heads state_channels"],
        output_projection: Float[Array, "suffix_tokens query_key_heads state_channels"],
        time_delta_log: Float[Array, "suffix_tokens value_heads"],
        initial_state: Float[Array, "value_heads head_channels state_channels"],
    ) -> tuple[
        Float[Array, "suffix_tokens value_heads head_channels"],
        Float[Array, "value_heads head_channels state_channels"],
    ]:
        def scan_fn(
            carry_state: Float[Array, "value_heads head_channels state_channels"],
            step_inputs: tuple[
                Float[Array, "value_heads head_channels"],
                Float[Array, "query_key_heads state_channels"],
                Float[Array, "query_key_heads state_channels"],
                Float[Array, " value_heads"],
            ],
        ) -> tuple[
            Float[Array, "value_heads head_channels state_channels"],
            Float[Array, "value_heads head_channels"],
        ]:
            hidden_state_t, input_proj_t, output_proj_t, time_delta_log_t = step_inputs

            time_delta_t = jax.nn.softplus(time_delta_log_t)[:, None]

            heads_per_group = self.num_value_heads // self.num_query_key_heads
            expanded_input_proj = jnp.repeat(input_proj_t, heads_per_group, axis=0)
            expanded_output_proj = jnp.repeat(output_proj_t, heads_per_group, axis=0)

            normalized_hidden = hidden_state_t / (time_delta_t + 1e-8)

            decay_factor = jnp.exp(-time_delta_t)[:, :, None]

            input_contribution = (
                time_delta_t[:, :, None] * normalized_hidden[:, :, None] * expanded_input_proj[:, None, :]
            )
            updated_state = decay_factor * carry_state + input_contribution

            output_t = einsum(
                updated_state,
                expanded_output_proj,
                "value_heads head_channels state_channels, value_heads state_channels -> value_heads head_channels",
            )

            return updated_state, output_t

        final_state, outputs = jax.lax.scan(
            scan_fn,
            initial_state,
            (hidden_states, input_projection, output_projection, time_delta_log),
        )

        return outputs, final_state

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: MambaStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> Mamba2Result:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for Mamba2.")

        num_tokens = inputs.shape[0]

        padded_length = ((num_tokens + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        padding_amount = padded_length - num_tokens
        padded_inputs = jnp.pad(inputs, ((0, padding_amount), (0, 0)), mode="constant", constant_values=0)

        conv_inputs, gate_values, time_delta_log = vmap(self.in_projection)(padded_inputs)

        conv_state_input = state.conv_state if state is not None else None
        (hidden_states, input_projection, output_projection), updated_conv_state = self.conv(
            conv_inputs,
            conv_state_input,
        )

        hidden_states = self.config.activation(hidden_states)
        input_projection = self.config.activation(input_projection)
        output_projection = self.config.activation(output_projection)

        hidden_states = rearrange(
            hidden_states,
            "suffix_tokens (value_heads head_channels) -> suffix_tokens value_heads head_channels",
            value_heads=self.num_value_heads,
        )
        input_projection = rearrange(
            input_projection,
            "suffix_tokens (query_key_heads state_channels) -> suffix_tokens query_key_heads state_channels",
            query_key_heads=self.num_query_key_heads,
        )
        output_projection = rearrange(
            output_projection,
            "suffix_tokens (query_key_heads state_channels) -> suffix_tokens query_key_heads state_channels",
            query_key_heads=self.num_query_key_heads,
        )
        time_delta_log = rearrange(
            time_delta_log,
            "suffix_tokens value_heads -> suffix_tokens value_heads",
            value_heads=self.num_value_heads,
        )

        if state is not None:
            current_ssm_state = state.ssm_state
        else:
            current_ssm_state = jnp.zeros(
                (self.num_value_heads, self.head_dim, self.state_dim),
                dtype=hidden_states.dtype,
            )

        ssm_outputs, final_ssm_state = self._chunk_scan(
            hidden_states,
            input_projection,
            output_projection,
            time_delta_log,
            current_ssm_state,
        )

        skip_contribution = einsum(
            self.skip_connection_weight,
            hidden_states,
            "value_heads, suffix_tokens value_heads head_channels -> suffix_tokens value_heads head_channels",
        )
        ssm_outputs = ssm_outputs + skip_contribution

        ssm_outputs = rearrange(
            ssm_outputs,
            "suffix_tokens value_heads head_channels -> suffix_tokens (value_heads head_channels)",
        )

        gated_outputs = ssm_outputs * self.config.activation(gate_values + self.gate_bias)

        (outputs,) = vmap(self.out_projection)(gated_outputs)

        outputs = outputs[:num_tokens]

        if return_updated_state:
            if length_without_padding is not None:
                if conv_state_input is not None:
                    prefix = conv_state_input
                else:
                    _, num_conv_channels = conv_inputs.shape
                    prefix = jnp.zeros((self.conv.kernel_size, num_conv_channels), dtype=conv_inputs.dtype)

                combined = jnp.concatenate([prefix, conv_inputs], axis=0)

                _, num_conv_channels = combined.shape
                final_conv_state = jax.lax.dynamic_slice(
                    combined,
                    (length_without_padding, 0),
                    (self.conv.kernel_size, num_conv_channels),
                )
            else:
                final_conv_state = updated_conv_state

            updated_state = MambaStateLayer.init(
                conv_state=final_conv_state,
                ssm_state=final_ssm_state,
            )
        else:
            updated_state = None

        return Mamba2Result(
            outputs=outputs,
            state=updated_state,
        )

    def init_static_state(self, capacity: int) -> MambaStateLayer:  # noqa: ARG002
        conv_channels = self.inner_dim + 2 * self.num_query_key_heads * self.state_dim
        return MambaStateLayer.empty(
            self.conv.kernel_size,
            conv_channels,
            self.num_value_heads,
            self.head_dim,
            self.state_dim,
            self.activation_precision,
        )

    def export_weights(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = {
            "in_projection": self.in_projection.export_weights(),
            "out_projection": self.out_projection.export_weights(),
            "conv_weight": self.conv.weight,
            "skip_connection_weight": self.skip_connection_weight,
            "gate_bias": self.gate_bias,
        }
        if self.conv.bias is not None:
            result["conv_bias"] = self.conv.bias
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["in_projection"], Mapping)
        assert isinstance(weights["out_projection"], Mapping)
        assert isinstance(weights["conv_weight"], Array)
        assert isinstance(weights["skip_connection_weight"], Array)
        assert isinstance(weights["gate_bias"], Array)

        if self.conv.bias is not None:
            assert isinstance(weights["conv_bias"], Array)
            conv_bias = weights["conv_bias"]
        else:
            conv_bias = None

        conv = CausalConv1d(
            weight=weights["conv_weight"],
            bias=conv_bias,
            output_dims=self.conv.output_dims,
            kernel_size=self.conv.kernel_size,
        )

        return replace(
            self,
            in_projection=self.in_projection.import_weights(weights["in_projection"]),
            out_projection=self.out_projection.import_weights(weights["out_projection"]),
            conv=conv,
            skip_connection_weight=weights["skip_connection_weight"],
            gate_bias=weights["gate_bias"],
        )
