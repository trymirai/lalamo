from collections.abc import Mapping
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
    "CausalConv1d",
    "CausalConv1dConfig",
    "Mamba2",
    "Mamba2Config",
    "Mamba2Result",
]


Mamba2Result = TokenMixerResult[MambaStateLayer]


@dataclass(frozen=True)
class CausalConv1dConfig:
    kernel_size: int
    activation: Activation
    precision: DTypeLike
    has_biases: bool

    def random_init(self, channels: int, *, key: PRNGKeyArray) -> "CausalConv1d":
        weight = jax.random.uniform(
            key,
            (channels, self.kernel_size),
            minval=-1.0,
            maxval=1.0,
            dtype=self.precision,
        )
        if self.has_biases:
            bias = jnp.zeros((channels,), dtype=self.precision)
        else:
            bias = None
        return CausalConv1d(self, weight=weight, bias=bias)

    def empty(self, channels: int) -> "CausalConv1d":
        weight = dummy_array((channels, self.kernel_size), self.precision)
        if self.has_biases:
            bias = dummy_array((channels,), self.precision)
        else:
            bias = None
        return CausalConv1d(self, weight=weight, bias=bias)


class CausalConv1d(eqx.Module):
    config: CausalConv1dConfig
    weight: Float[Array, "channels kernel"]
    bias: Float[Array, " channels"] | None

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        state: Float[Array, "history channels"] | None = None,
    ) -> tuple[Float[Array, "suffix_tokens channels"], Float[Array, "history channels"]]:
        num_tokens, num_channels = inputs.shape

        if state is None:
            state = jnp.zeros((self.config.kernel_size, num_channels), dtype=inputs.dtype)

        padded = jnp.concatenate([state, inputs], axis=0)

        def apply_kernel(idx: Int[Array, ""]) -> Float[Array, " channels"]:
            window = jax.lax.dynamic_slice(padded, (idx, 0), (self.config.kernel_size, num_channels))
            result = einsum(window, self.weight, "kernel channels, channels kernel -> channels")
            if self.bias is not None:
                result = result + self.bias
            return result

        conv_outputs = vmap(apply_kernel)(jnp.arange(num_tokens))
        updated_state = jax.lax.dynamic_slice(padded, (num_tokens, 0), (self.config.kernel_size, num_channels))
        activated = self.config.activation(conv_outputs)
        return activated, updated_state

    def export_weights(self) -> ParameterTree:
        result: dict[str, Array] = {"weight": self.weight}
        if self.bias is not None:
            result["bias"] = self.bias
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> "CausalConv1d":
        assert isinstance(weights, Mapping)
        assert isinstance(weights["weight"], Array)
        if self.bias is not None:
            assert isinstance(weights["bias"], Array)
            bias = weights["bias"]
        else:
            bias = None
        return replace(
            self,
            weight=weights["weight"],
            bias=bias,
        )


@dataclass(frozen=True)
class Mamba2Config(TokenMixerConfigBase):
    in_projection_config: LinearConfig
    out_projection_config: LinearConfig
    conv_config: CausalConv1dConfig

    num_value_heads: int
    num_groups: int
    head_dim: int
    state_dim: int
    expand: int

    has_in_biases: bool
    has_out_biases: bool

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
                self.inner_dim + 2 * self.num_groups * self.state_dim,
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

        conv_channels = self.inner_dim + 2 * self.num_groups * self.state_dim
        conv = self.conv_config.random_init(conv_channels, key=conv_key)

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
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            state_dim=self.state_dim,
        )

    def empty(
        self,
        model_dim: int,
    ) -> "Mamba2":
        in_projection = self.in_projection_config.empty(
            input_dim=model_dim,
            output_dims=(
                self.inner_dim + 2 * self.num_groups * self.state_dim,
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

        conv_channels = self.inner_dim + 2 * self.num_groups * self.state_dim
        conv = self.conv_config.empty(conv_channels)

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
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            state_dim=self.state_dim,
        )


class Mamba2(TokenMixerBase[Mamba2Config, MambaStateLayer]):
    in_projection: LinearBase
    conv: CausalConv1d
    out_projection: LinearBase

    skip_connection_weight: Float[Array, " value_heads"]
    gate_bias: Float[Array, " inner_dim"]

    num_value_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

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
        if self.num_value_heads % self.num_groups != 0:
            raise ValueError(
                f"Number of value heads ({self.num_value_heads}) must be divisible by "
                f"number of groups ({self.num_groups})",
            )

    def _scan(
        self,
        hidden_states: Float[Array, "suffix_tokens value_heads head_channels"],
        input_projection: Float[Array, "suffix_tokens groups state_channels"],
        output_projection: Float[Array, "suffix_tokens groups state_channels"],
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
                Float[Array, "groups state_channels"],
                Float[Array, "groups state_channels"],
                Float[Array, " value_heads"],
            ],
        ) -> tuple[
            Float[Array, "value_heads head_channels state_channels"],
            Float[Array, "value_heads head_channels"],
        ]:
            hidden_state_t, input_proj_t, output_proj_t, time_delta_log_t = step_inputs
            dt = jax.nn.softplus(time_delta_log_t)[:, None]
            heads_per_group = self.num_value_heads // self.num_groups

            # Group heads without materializing repeated B/C
            hidden_grouped = rearrange(
                hidden_state_t,
                "(groups heads) head_channels -> groups heads head_channels",
                groups=self.num_groups,
                heads=heads_per_group,
            )
            x_norm_grouped = hidden_grouped / (
                dt.reshape(self.num_value_heads)[
                    rearrange(
                        jnp.arange(self.num_value_heads),
                        "(groups heads)-> groups heads",
                        groups=self.num_groups,
                        heads=heads_per_group,
                    )
                ][:, :, None]
                + 1e-8
            )

            # decay and mix per value head -> reshape per group
            decay = jnp.exp(-dt)[:, :, None]
            mix = dt[:, :, None]
            decay_group = rearrange(
                decay,
                "(groups heads) 1 1 -> groups heads 1 1",
                groups=self.num_groups,
                heads=heads_per_group,
            )
            mix_group = rearrange(
                mix,
                "(groups heads) 1 1 -> groups heads 1 1",
                groups=self.num_groups,
                heads=heads_per_group,
            )

            # B/C are per group: broadcast over heads without repeating tensors
            input_contribution_group = mix_group * x_norm_grouped[:, :, :, None] * input_proj_t[:, None, None, :]
            carry_state_group = rearrange(
                carry_state,
                "(groups heads) head_channels state_channels -> groups heads head_channels state_channels",
                groups=self.num_groups,
                heads=heads_per_group,
            )
            updated_state_group = decay_group * carry_state_group + input_contribution_group

            # Output using group C
            output_group = einsum(
                updated_state_group,
                output_proj_t,
                "groups heads head_channels state_channels, groups state_channels -> groups heads head_channels",
            )
            updated_state = rearrange(
                updated_state_group,
                "groups heads head_channels state_channels -> (groups heads) head_channels state_channels",
            )
            output_t = rearrange(output_group, "groups heads head_channels -> (groups heads) head_channels")

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

        conv_inputs, gate_values, time_delta_log = vmap(self.in_projection)(inputs)

        conv_state_input = state.conv_state if state is not None else None
        conv_activated, updated_conv_state = self.conv(
            conv_inputs,
            conv_state_input,
        )

        x_channels, input_proj_channels, output_proj_channels = jnp.split(
            conv_activated,
            [
                self.inner_dim,
                self.inner_dim + self.num_groups * self.state_dim,
            ],
            axis=-1,
        )

        hidden_states = rearrange(
            x_channels,
            "suffix_tokens (value_heads head_channels) -> suffix_tokens value_heads head_channels",
            value_heads=self.num_value_heads,
        )
        input_projection = rearrange(
            input_proj_channels,
            "suffix_tokens (groups state_channels) -> suffix_tokens groups state_channels",
            groups=self.num_groups,
        )
        output_projection = rearrange(
            output_proj_channels,
            "suffix_tokens (groups state_channels) -> suffix_tokens groups state_channels",
            groups=self.num_groups,
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

        ssm_outputs, final_ssm_state = self._scan(
            hidden_states,
            input_projection,
            output_projection,
            time_delta_log,
            current_ssm_state,
        )

        skip_contribution = self.skip_connection_weight[None, :, None] * hidden_states
        ssm_outputs = ssm_outputs + skip_contribution

        ssm_outputs = rearrange(
            ssm_outputs,
            "suffix_tokens value_heads head_channels -> suffix_tokens (value_heads head_channels)",
        )

        gated_outputs = ssm_outputs * jax.nn.silu(gate_values + self.gate_bias)

        (outputs,) = vmap(self.out_projection)(gated_outputs)

        if return_updated_state:
            if state is not None:
                if length_without_padding is not None:
                    updated_state = state.extend(conv_inputs, final_ssm_state, added_length=length_without_padding)
                else:
                    updated_state = state.extend(conv_inputs, final_ssm_state)
            else:
                updated_state = MambaStateLayer.init(
                    conv_state=updated_conv_state,
                    ssm_state=final_ssm_state,
                )
        else:
            updated_state = None

        return Mamba2Result(
            outputs=outputs,
            state=updated_state,
        )

    def init_static_state(self, capacity: int) -> MambaStateLayer:  # noqa: ARG002
        conv_channels = self.inner_dim + 2 * self.num_groups * self.state_dim
        return MambaStateLayer.empty(
            self.conv.config.kernel_size - 1,
            conv_channels,
            self.num_value_heads,
            self.head_dim,
            self.state_dim,
            self.activation_precision,
        )

    def export_weights(self) -> ParameterTree:
        return {
            "in_projection": self.in_projection.export_weights(),
            "out_projection": self.out_projection.export_weights(),
            "conv": self.conv.export_weights(),
            "skip_connection_weight": self.skip_connection_weight,
            "gate_bias": self.gate_bias,
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["in_projection"], Mapping)
        assert isinstance(weights["out_projection"], Mapping)
        assert isinstance(weights["conv"], Mapping)
        assert isinstance(weights["skip_connection_weight"], Array)
        assert isinstance(weights["gate_bias"], Array)

        return replace(
            self,
            in_projection=self.in_projection.import_weights(weights["in_projection"]),
            out_projection=self.out_projection.import_weights(weights["out_projection"]),
            conv=self.conv.import_weights(weights["conv"]),
            skip_connection_weight=weights["skip_connection_weight"],
            gate_bias=weights["gate_bias"],
        )
