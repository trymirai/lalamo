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
    "exp_segsum",
]


def exp_segsum(x: Float[Array, "... T"]) -> Float[Array, "... T T"]:
    """Compute exp(segsum(x)) as lower-triangular matrix using cumsum difference."""
    seq_len = x.shape[-1]
    cs = jnp.cumsum(x, axis=-1)
    diff = cs[..., :, None] - cs[..., None, :]
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return jnp.where(mask, jnp.exp(diff), 0.0)


def fused_ssd_intra_chunk(
    a_cumsum: Float[Array, "groups heads_per_group chunks chunk_size"],
    cb: Float[Array, "chunks chunk_size chunk_size groups"],
    x: Float[Array, "chunks chunk_size groups heads_per_group head_dim"],
) -> Float[Array, "chunks chunk_size groups heads_per_group head_dim"]:
    """Compute intra-chunk diagonal block outputs for SSD.

    Avoids materializing the full global L matrix by computing decay locally per (chunk, group, head).
    """
    groups, heads_per_group, chunks, chunk_size = a_cumsum.shape

    def compute_one(
        a_cs: Float[Array, " chunk_size"],
        cb_slice: Float[Array, "chunk_size chunk_size"],
        x_slice: Float[Array, "chunk_size head_dim"],
    ) -> Float[Array, "chunk_size head_dim"]:
        diff = a_cs[:, None] - a_cs[None, :]
        mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
        decay_local = jnp.where(mask, jnp.exp(diff), 0.0)
        weighted = decay_local * cb_slice
        return weighted @ x_slice

    def compute_chunk_group_head(c: int, g: int, r: int) -> Float[Array, "chunk_size head_dim"]:
        return compute_one(a_cumsum[g, r, c, :], cb[c, :, :, g], x[c, :, g, r, :])

    result = jax.vmap(
        lambda c: jax.vmap(
            lambda g: jax.vmap(
                lambda r: compute_chunk_group_head(c, g, r),
            )(jnp.arange(heads_per_group)),
        )(jnp.arange(groups)),
    )(jnp.arange(chunks))

    return jnp.transpose(result, (0, 3, 1, 2, 4))


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

    chunk_size: int = 256

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

    def _step(
        self,
        x: Float[Array, "heads head_dim"],
        b: Float[Array, "groups state_dim"],
        c: Float[Array, "groups state_dim"],
        dt_log: Float[Array, " heads"],
        state: Float[Array, "heads head_dim state_dim"],
    ) -> tuple[Float[Array, "heads head_dim"], Float[Array, "heads head_dim state_dim"]]:
        """Single-token SSM state update without scan overhead."""
        heads_per_group = self.num_heads // self.num_groups

        dt = jax.nn.softplus(dt_log)
        decay = jnp.exp(-dt)[:, None, None]
        mix = dt[:, None, None]

        b_expanded = jnp.repeat(b, heads_per_group, axis=0)
        c_expanded = jnp.repeat(c, heads_per_group, axis=0)
        x_norm = x / (dt[:, None] + 1e-8)

        input_contribution = mix * x_norm[:, :, None] * b_expanded[:, None, :]
        new_state = decay * state + input_contribution
        output = einsum(new_state, c_expanded, "h p n, h n -> h p")

        return output, new_state

    def _conv_step(
        self,
        x: Float[Array, " channels"],
        state: Float[Array, "kernel_minus_1 channels"],
    ) -> tuple[Float[Array, " channels"], Float[Array, "kernel_minus_1 channels"]]:
        """Single-token conv update without full convolution."""
        full_input = jnp.concatenate([state, x[None, :]], axis=0)
        output = einsum(full_input, self.conv.weights, "k c, c k -> c")
        if self.conv.biases is not None:
            output = output + self.conv.biases
        new_state = jnp.concatenate([state[1:], x[None, :]], axis=0)
        return output, new_state

    def _decode_step(
        self,
        inputs: Float[Array, "1 channels"],
        state: Mamba2StateLayer,
    ) -> Mamba2Result:
        """Optimized path for single-token decode without scan machinery."""
        x = inputs[0]

        conv_in, gate, dt_log = self.in_projection(x)
        conv_out, new_conv_state = self._conv_step(conv_in, state.conv_state)
        conv_activated = self.config.activation(conv_out)

        x_ssm, b_flat, c_flat = jnp.split(
            conv_activated,
            [self.inner_dim, self.inner_dim + self.num_groups * self.state_dim],
        )
        x_ssm = rearrange(x_ssm, "(h p) -> h p", h=self.num_heads)
        b = rearrange(b_flat, "(g n) -> g n", g=self.num_groups)
        c = rearrange(c_flat, "(g n) -> g n", g=self.num_groups)

        y, new_ssm_state = self._step(x_ssm, b, c, dt_log, state.ssm_state)

        y = y + self.skip_connection_weight[:, None] * x_ssm
        y = rearrange(y, "h p -> (h p)")
        gated = y * jax.nn.silu(gate + self.gate_bias)
        (output,) = self.out_projection(gated)

        return Mamba2Result(
            outputs=output[None, :],
            state=Mamba2StateLayer(new_conv_state, new_ssm_state),
        )

    def _chunked_scan(
        self,
        x: Float[Array, "seq heads head_dim"],
        b: Float[Array, "seq groups state_dim"],
        c: Float[Array, "seq groups state_dim"],
        dt: Float[Array, "seq heads"],
        initial_state: Float[Array, "heads head_dim state_dim"],
        chunk_size: int,
        num_steps: Int[Array, ""] | int,
        d: Float[Array, " heads"] | None = None,
        z: Float[Array, "seq heads head_dim"] | None = None,
        z_bias: Float[Array, "heads head_dim"] | None = None,
    ) -> tuple[Float[Array, "seq heads head_dim"], Float[Array, "heads head_dim state_dim"]]:
        """Chunked parallel scan implementing the SSD algorithm."""
        seq_len = x.shape[0]
        num_steps = jnp.asarray(num_steps, dtype=jnp.int32)

        pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_len > 0:
            x = jnp.pad(x, ((0, pad_len), (0, 0), (0, 0)))
            b = jnp.pad(b, ((0, pad_len), (0, 0), (0, 0)))
            c = jnp.pad(c, ((0, pad_len), (0, 0), (0, 0)))
            dt = jnp.pad(dt, ((0, pad_len), (0, 0)))
            if z is not None:
                z = jnp.pad(z, ((0, pad_len), (0, 0), (0, 0)))

        x_orig = x
        b_orig = b
        dt_orig = dt

        padded_len = x.shape[0]
        position_indices = jnp.arange(padded_len)
        valid_mask = (position_indices < num_steps).astype(x.dtype)
        x = x * valid_mask[:, None, None]
        b = b * valid_mask[:, None, None]

        x = rearrange(x, "(c l) (g r) p -> c l g r p", l=chunk_size, g=self.num_groups)
        a = rearrange(-dt, "(c l) (g r) -> g r c l", l=chunk_size, g=self.num_groups)
        b = rearrange(b, "(c l) g n -> c l g n", l=chunk_size)
        c = rearrange(c, "(c l) g n -> c l g n", l=chunk_size)
        a_cumsum = jnp.cumsum(a, axis=-1)

        cb = einsum(c, b, "c l g n, c s g n -> c l s g")
        y_diag = fused_ssd_intra_chunk(a_cumsum, cb, x)

        decay_states = jnp.exp(a_cumsum[:, :, :, -1:] - a_cumsum)
        states = einsum(b, decay_states, x, "c l g n, g r c l, c l g r p -> c g r p n")

        initial_state_grouped = rearrange(initial_state, "(g r) p n -> g r p n", g=self.num_groups)
        states = jnp.concatenate([initial_state_grouped[None, ...], states], axis=0)
        a_chunk_ends = jnp.pad(a_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0)))
        decay_chunk = exp_segsum(a_chunk_ends)
        new_states = einsum(decay_chunk, states, "g r z c, c g r p n -> z g r p n")
        states = new_states[:-1]

        state_decay_out = jnp.exp(a_cumsum)
        y_off = einsum(c, states, state_decay_out, "c l g n, c g r p n, g r c l -> c l g r p")

        y = y_diag + y_off
        if d is not None:
            d_grouped = rearrange(d, "(g r) -> g r", g=self.num_groups)
            y = y + d_grouped[None, None, :, :, None] * x
        y = rearrange(y, "c l g r p -> (c l) (g r) p")

        if z is not None:
            gate = z + z_bias[None, :, :] if z_bias is not None else z
            y = y * jax.nn.silu(gate)

        y = y[:seq_len]

        new_states_flat = rearrange(new_states, "c g r p n -> c (g r) p n")
        final_state = self._compute_final_state(x_orig, b_orig, dt_orig, new_states_flat, num_steps, chunk_size)

        return y, final_state

    def _compute_final_state(
        self,
        x: Float[Array, "seq heads head_dim"],
        b: Float[Array, "seq groups state_dim"],
        dt: Float[Array, "seq heads"],
        chunk_states: Float[Array, "chunks_plus_1 heads head_dim state_dim"],
        num_steps: Int[Array, ""],
        chunk_size: int,
    ) -> Float[Array, "heads head_dim state_dim"]:
        """Compute the exact final state at position num_steps using precomputed chunk_states."""
        heads_per_group = self.num_heads // self.num_groups

        chunk_idx = num_steps // chunk_size
        pos_in_chunk = num_steps % chunk_size
        chunk_start_state = jax.lax.dynamic_index_in_dim(chunk_states, chunk_idx, axis=0, keepdims=False)

        def at_boundary() -> Float[Array, "heads head_dim state_dim"]:
            return chunk_start_state

        def within_chunk() -> Float[Array, "heads head_dim state_dim"]:
            chunk_start_pos = chunk_idx * chunk_size
            x_chunk = jax.lax.dynamic_slice(x, (chunk_start_pos, 0, 0), (chunk_size, x.shape[1], x.shape[2]))
            b_chunk = jax.lax.dynamic_slice(b, (chunk_start_pos, 0, 0), (chunk_size, b.shape[1], b.shape[2]))
            dt_chunk = jax.lax.dynamic_slice(dt, (chunk_start_pos, 0), (chunk_size, dt.shape[1]))

            a_cumsum = jnp.cumsum(-dt_chunk, axis=0)
            last_pos_idx = pos_in_chunk - 1
            a_cumsum_at_last = jax.lax.dynamic_index_in_dim(a_cumsum, last_pos_idx, axis=0, keepdims=False)

            decayed_start = jnp.exp(a_cumsum_at_last)[:, None, None] * chunk_start_state

            decay_to_last = jnp.exp(a_cumsum_at_last[None, :] - a_cumsum)
            mask = jnp.arange(chunk_size) <= last_pos_idx
            masked_decay = jnp.where(mask[:, None], decay_to_last, 0.0)
            b_expanded = jnp.repeat(b_chunk, heads_per_group, axis=1)
            input_contrib = einsum(masked_decay, b_expanded, x_chunk, "l h, l h n, l h p -> h p n")

            return decayed_start + input_contrib

        return jax.lax.cond(pos_in_chunk == 0, at_boundary, within_chunk)

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

        seq_len, _ = inputs.shape

        if seq_len == 1 and return_updated_state:
            return self._decode_step(inputs, state)

        conv_inputs, gate_values, time_delta_log = vmap(self.in_projection)(inputs)

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

        if length_without_padding is None:
            length_without_padding, _ = inputs.shape

        gate_values_reshaped = rearrange(
            gate_values,
            "suffix_tokens (heads head_channels) -> suffix_tokens heads head_channels",
            heads=self.num_heads,
        )
        gate_bias_reshaped = rearrange(
            self.gate_bias,
            "(heads head_channels) -> heads head_channels",
            heads=self.num_heads,
        )

        dt = jax.nn.softplus(time_delta_log)
        ssm_outputs, final_ssm_state = self._chunked_scan(
            hidden_states,
            input_projection,
            output_projection,
            dt,
            state.ssm_state,
            self.config.chunk_size,
            length_without_padding,
            d=self.skip_connection_weight,
            z=gate_values_reshaped,
            z_bias=gate_bias_reshaped,
        )

        ssm_outputs_flat = rearrange(
            ssm_outputs,
            "suffix_tokens heads head_channels -> suffix_tokens (heads head_channels)",
        )
        (outputs,) = vmap(self.out_projection)(ssm_outputs_flat)

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
