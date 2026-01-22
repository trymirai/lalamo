import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_array, require_tree
from lalamo.modules.activations import Activation
from lalamo.modules.common import LalamoModule, PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.rope import PositionalEmbeddings

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .state import Mamba2StateLayer

__all__ = [
    "Mamba2",
    "Mamba2Config",
    "Mamba2Result",
    "SeparableCausalConv",
    "SeparableCausalConvConfig",
]


Mamba2Result = TokenMixerResult[Mamba2StateLayer]


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


@dataclass(frozen=True)
class Mamba2Config(TokenMixerConfigBase):
    in_projection_config: LinearConfig
    out_projection_config: LinearConfig
    conv_config: SeparableCausalConvConfig
    activation: Activation

    kernel_size: int
    num_heads: int
    num_groups: int
    head_dim: int
    state_dim: int
    expansion_factor: int

    has_in_biases: bool
    has_out_biases: bool

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def rope_dim(self) -> None:
        return None

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
                self.num_heads,
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
        conv = self.conv_config.random_init(conv_channels, self.kernel_size, key=conv_key)

        skip_connection_weight = jax.random.normal(
            skip_key,
            (self.num_heads,),
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
            num_heads=self.num_heads,
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
                self.num_heads,
            ),
            has_biases=self.has_in_biases,
        )

        out_projection = self.out_projection_config.empty(
            self.inner_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
        )

        conv_channels = self.inner_dim + 2 * self.num_groups * self.state_dim
        conv = self.conv_config.empty(conv_channels, self.kernel_size)

        skip_connection_weight = dummy_array((self.num_heads,), in_projection.activation_precision)
        gate_bias = dummy_array((self.inner_dim,), in_projection.activation_precision)

        return Mamba2(
            self,
            in_projection=in_projection,
            conv=conv,
            out_projection=out_projection,
            skip_connection_weight=skip_connection_weight,
            gate_bias=gate_bias,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            state_dim=self.state_dim,
        )


class Mamba2(TokenMixerBase[Mamba2Config, Mamba2StateLayer]):
    in_projection: LinearBase
    conv: SeparableCausalConv
    out_projection: LinearBase

    skip_connection_weight: Float[Array, " heads"]
    gate_bias: Float[Array, " inner_channels"]

    num_heads: int = eqx.field(static=True)
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
        return self.num_heads * self.head_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return PositionalEmbeddingSelector.NONE

    def __post_init__(self) -> None:
        if self.skip_connection_weight.shape != (self.num_heads,):
            raise ValueError(
                f"Skip connection weight must have shape (num_heads,) = ({self.num_heads},), "
                f"got {self.skip_connection_weight.shape}",
            )
        if self.gate_bias.shape != (self.inner_dim,):
            raise ValueError(
                f"Gate bias must have shape (inner_dim,) = ({self.inner_dim},), got {self.gate_bias.shape}",
            )
        if self.num_heads % self.num_groups != 0:
            raise ValueError(
                f"Number of value heads ({self.num_heads}) must be divisible by number of groups ({self.num_groups})",
            )

    def _scan(
        self,
        hidden_states: Float[Array, "suffix_tokens heads head_channels"],
        input_projection: Float[Array, "suffix_tokens groups state_channels"],
        output_projection: Float[Array, "suffix_tokens groups state_channels"],
        time_delta_log: Float[Array, "suffix_tokens heads"],
        initial_state: Float[Array, "heads head_channels state_channels"],
        num_steps: Int[Array, ""] | int,
    ) -> tuple[
        Float[Array, "suffix_tokens heads head_channels"],
        Float[Array, "heads head_channels state_channels"],
    ]:
        def scan_fn(
            index_and_carry_state: tuple[Int[Array, ""], Float[Array, "heads head_channels state_channels"]],
            step_inputs: tuple[
                Float[Array, "heads head_channels"],
                Float[Array, "groups state_channels"],
                Float[Array, "groups state_channels"],
                Float[Array, " heads"],
            ],
        ) -> tuple[
            tuple[Int[Array, ""], Float[Array, "heads head_channels state_channels"]],
            Float[Array, "heads head_channels"],
        ]:
            index, carry_state = index_and_carry_state
            hidden_state_t, input_proj_t, output_proj_t, time_delta_log_t = step_inputs
            dt = jax.nn.softplus(time_delta_log_t)[:, None]
            heads_per_group = self.num_heads // self.num_groups

            hidden_grouped = rearrange(
                hidden_state_t,
                "(groups heads) head_channels -> groups heads head_channels",
                groups=self.num_groups,
                heads=heads_per_group,
            )
            x_norm_grouped = hidden_grouped / (
                dt.reshape(self.num_heads)[
                    rearrange(
                        jnp.arange(self.num_heads),
                        "(groups heads)-> groups heads",
                        groups=self.num_groups,
                        heads=heads_per_group,
                    )
                ][:, :, None]
                + 1e-8
            )

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

            input_contribution_group = mix_group * x_norm_grouped[:, :, :, None] * input_proj_t[:, None, None, :]
            carry_state_group = rearrange(
                carry_state,
                "(groups heads) head_channels state_channels -> groups heads head_channels state_channels",
                groups=self.num_groups,
                heads=heads_per_group,
            )
            updated_state_group = decay_group * carry_state_group + input_contribution_group

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

            propagated_state = jax.lax.cond(index < num_steps, lambda: updated_state, lambda: carry_state)

            return (index + 1, propagated_state), output_t

        (_, final_state), outputs = jax.lax.scan(
            scan_fn,
            (jnp.zeros((), dtype=jnp.int32), initial_state),
            (hidden_states, input_projection, output_projection, time_delta_log),
        )

        return outputs, final_state

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: Mamba2StateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> Mamba2Result:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for Mamba2.")

        conv_inputs, gate_values, time_delta_log = vmap(self.in_projection)(inputs)

        if state is None:
            state = Mamba2StateLayer.init(
                self.config.kernel_size,
                self.inner_dim,
                self.num_heads,
                self.num_groups,
                self.head_dim,
                self.state_dim,
                self.activation_precision,
            )

        conv_output, updated_conv_state = self.conv(
            conv_inputs,
            length_without_padding,
            state.conv_state,
            return_updated_state=return_updated_state,
        )
        conv_activated = self.config.activation(conv_output)

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
            "suffix_tokens (heads head_channels) -> suffix_tokens heads head_channels",
            heads=self.num_heads,
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
            "suffix_tokens heads -> suffix_tokens heads",
            heads=self.num_heads,
        )

        if length_without_padding is None:
            length_without_padding, _ = inputs.shape

        ssm_outputs, final_ssm_state = self._scan(
            hidden_states,
            input_projection,
            output_projection,
            time_delta_log,
            state.ssm_state,
            length_without_padding,
        )

        skip_contribution = self.skip_connection_weight[None, :, None] * hidden_states
        ssm_outputs = ssm_outputs + skip_contribution

        ssm_outputs = rearrange(
            ssm_outputs,
            "suffix_tokens heads head_channels -> suffix_tokens (heads head_channels)",
        )

        gated_outputs = ssm_outputs * jax.nn.silu(gate_values + self.gate_bias)

        (outputs,) = vmap(self.out_projection)(gated_outputs)

        if return_updated_state:
            assert updated_conv_state is not None
            updated_state = Mamba2StateLayer(updated_conv_state, final_ssm_state)
        else:
            updated_state = None

        return Mamba2Result(
            outputs=outputs,
            state=updated_state,
        )

    def init_static_state(self, capacity: int) -> Mamba2StateLayer:  # noqa: ARG002
        return Mamba2StateLayer.init(
            self.config.kernel_size,
            self.inner_dim,
            self.num_heads,
            self.num_groups,
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

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            in_projection=self.in_projection.import_weights(require_tree(weights["in_projection"])),
            out_projection=self.out_projection.import_weights(require_tree(weights["out_projection"])),
            conv=self.conv.import_weights(require_tree(weights["conv"])),
            skip_connection_weight=require_array(weights["skip_connection_weight"]),
            gate_bias=require_array(weights["gate_bias"]),
        )
