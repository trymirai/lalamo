from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_array, require_tree
from lalamo.modules.activations import Activation
from lalamo.modules.common import PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixers.state.ssm_state import SSMStateLayer

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .convolutions import SeparableCausalConv, SeparableCausalConvConfig

__all__ = [
    "Mamba2",
    "Mamba2Config",
    "Mamba2Result",
    "SeparableCausalConv",
    "SeparableCausalConvConfig",
    "exp_segsum",
    "fused_ssd_intra_chunk",
]

Mamba2Result = TokenMixerResult[SSMStateLayer]
 
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

    def compute_chunk_group_head(chunk_idx: int, group_idx: int, head_idx: int) -> Float[Array, "chunk_size head_dim"]:
        return compute_one(
            a_cumsum[group_idx, head_idx, chunk_idx, :],
            cb[chunk_idx, :, :, group_idx],
            x[chunk_idx, :, group_idx, head_idx, :],
        )

    def over_heads(chunk_idx: int, group_idx: int) -> Float[Array, "heads_per_group chunk_size head_dim"]:
        return jax.vmap(lambda head_idx: compute_chunk_group_head(chunk_idx, group_idx, head_idx))(
            jnp.arange(heads_per_group),
        )

    def over_groups(chunk_idx: int) -> Float[Array, "groups heads_per_group chunk_size head_dim"]:
        return jax.vmap(lambda group_idx: over_heads(chunk_idx, group_idx))(jnp.arange(groups))

    result = jax.vmap(over_groups)(jnp.arange(chunks))

    return rearrange(
        result,
        "chunks groups heads_per_group chunk_size head_dim -> chunks chunk_size groups heads_per_group head_dim",
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
    def conv_dim(self) -> int:
        return self.inner_dim + 2 * self.num_groups * self.state_dim

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

        conv = self.conv_config.random_init(self.conv_dim, self.kernel_size, key=conv_key)

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

        conv = self.conv_config.empty(self.conv_dim, self.kernel_size)

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


class Mamba2(TokenMixerBase[Mamba2Config, SSMStateLayer]):
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
        values: Float[Array, "heads head_dim"],
        keys: Float[Array, "groups state_dim"],
        queries: Float[Array, "groups state_dim"],
        dt_log: Float[Array, " heads"],
        state: Float[Array, "heads head_dim state_dim"],
    ) -> tuple[Float[Array, "heads head_dim"], Float[Array, "heads head_dim state_dim"]]:
        """Single-token SSM state update without scan overhead."""
        heads_per_group = self.num_heads // self.num_groups

        dt = jax.nn.softplus(dt_log)
        decay = jnp.exp(-dt)[:, None, None]
        mix = dt[:, None, None]

        keys_expanded = jnp.repeat(keys, heads_per_group, axis=0)
        queries_expanded = jnp.repeat(queries, heads_per_group, axis=0)
        values_norm = values / (dt[:, None] + 1e-8)

        input_contribution = mix * values_norm[:, :, None] * keys_expanded[:, None, :]
        new_state = decay * state + input_contribution
        output = einsum(new_state, queries_expanded, "heads head_dim state_dim, heads state_dim -> heads head_dim")

        return output, new_state

    def _decode_step(
        self,
        inputs: Float[Array, "1 channels"],
        state: SSMStateLayer,
    ) -> Mamba2Result:
        """Optimized path for single-token decode without scan machinery."""
        token = inputs[0]

        conv_in, gate, dt_log = self.in_projection(token)
        conv_out, new_conv_state = self.conv.step(conv_in, state.conv_state)
        conv_activated = self.config.activation(conv_out)

        values_flat, input_proj_flat, output_proj_flat = jnp.split(
            conv_activated,
            [self.inner_dim, self.inner_dim + self.num_groups * self.state_dim],
        )
        values = rearrange(values_flat, "(heads head_dim) -> heads head_dim", heads=self.num_heads)
        keys = rearrange(input_proj_flat, "(groups state_dim) -> groups state_dim", groups=self.num_groups)
        queries = rearrange(output_proj_flat, "(groups state_dim) -> groups state_dim", groups=self.num_groups)

        y, new_ssm_state = self._step(values, keys, queries, dt_log, state.ssm_state)

        y = y + self.skip_connection_weight[:, None] * values
        y = rearrange(y, "heads head_dim -> (heads head_dim)")
        gated = y * jax.nn.silu(gate + self.gate_bias)
        (output,) = self.out_projection(gated)

        return Mamba2Result(
            outputs=output[None, :],
            state=SSMStateLayer(new_conv_state, new_ssm_state),
        )

    def _chunked_scan(
        self,
        values: Float[Array, "suffix_tokens heads head_dim"],
        keys: Float[Array, "suffix_tokens groups state_dim"],
        queries: Float[Array, "suffix_tokens groups state_dim"],
        dt: Float[Array, "suffix_tokens heads"],
        initial_state: Float[Array, "heads head_dim state_dim"],
        chunk_size: int,
        num_steps: Int[Array, ""] | int,
        d: Float[Array, " heads"] | None = None,
        z: Float[Array, "suffix_tokens heads head_dim"] | None = None,
        z_bias: Float[Array, "heads head_dim"] | None = None,
    ) -> tuple[Float[Array, "suffix_tokens heads head_dim"], Float[Array, "heads head_dim state_dim"]]:
        """Chunked parallel scan implementing the SSD algorithm."""
        seq_len = values.shape[0]
        num_steps = jnp.asarray(num_steps, dtype=jnp.int32)

        pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_len > 0:
            values = jnp.pad(values, ((0, pad_len), (0, 0), (0, 0)))
            keys = jnp.pad(keys, ((0, pad_len), (0, 0), (0, 0)))
            queries = jnp.pad(queries, ((0, pad_len), (0, 0), (0, 0)))
            dt = jnp.pad(dt, ((0, pad_len), (0, 0)))
            if z is not None:
                z = jnp.pad(z, ((0, pad_len), (0, 0), (0, 0)))

        values_orig = values
        keys_orig = keys
        dt_orig = dt

        padded_len = values.shape[0]
        position_indices = jnp.arange(padded_len)
        valid_mask = (position_indices < num_steps).astype(values.dtype)
        values = values * valid_mask[:, None, None]
        keys = keys * valid_mask[:, None, None]

        values = rearrange(
            values,
            "(chunks chunk_size) (groups heads_per_group) head_dim"
            " -> chunks chunk_size groups heads_per_group head_dim",
            chunk_size=chunk_size,
            groups=self.num_groups,
        )
        log_decay = rearrange(
            -dt,
            "(chunks chunk_size) (groups heads_per_group) -> groups heads_per_group chunks chunk_size",
            chunk_size=chunk_size,
            groups=self.num_groups,
        )
        keys_chunked = rearrange(
            keys,
            "(chunks chunk_size) groups state_dim -> chunks chunk_size groups state_dim",
            chunk_size=chunk_size,
        )
        queries_chunked = rearrange(
            queries,
            "(chunks chunk_size) groups state_dim -> chunks chunk_size groups state_dim",
            chunk_size=chunk_size,
        )
        log_decay_cumsum = jnp.cumsum(log_decay, axis=-1)

        queries_keys_prod = einsum(
            queries_chunked,
            keys_chunked,
            "chunks query_pos groups state_dim, chunks key_pos groups state_dim -> chunks query_pos key_pos groups",
        )
        y_diag = fused_ssd_intra_chunk(log_decay_cumsum, queries_keys_prod, values)

        decay_states = jnp.exp(log_decay_cumsum[:, :, :, -1:] - log_decay_cumsum)
        states = einsum(
            keys_chunked,
            decay_states,
            values,
            "chunks chunk_size groups state_dim, groups heads_per_group chunks chunk_size,"
            " chunks chunk_size groups heads_per_group head_dim"
            " -> chunks groups heads_per_group head_dim state_dim",
        )

        initial_state_grouped = rearrange(
            initial_state,
            "(groups heads_per_group) head_dim state_dim -> groups heads_per_group head_dim state_dim",
            groups=self.num_groups,
        )
        states = jnp.concatenate([initial_state_grouped[None, ...], states], axis=0)
        log_decay_chunk_ends = jnp.pad(log_decay_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0)))
        decay_chunk = exp_segsum(log_decay_chunk_ends)
        new_states = einsum(
            decay_chunk,
            states,
            "groups heads_per_group out_idx chunks,"
            " chunks groups heads_per_group head_dim state_dim"
            " -> out_idx groups heads_per_group head_dim state_dim",
        )
        states = new_states[:-1]

        state_decay_out = jnp.exp(log_decay_cumsum)
        y_off = einsum(
            queries_chunked,
            states,
            state_decay_out,
            "chunks chunk_size groups state_dim,"
            " chunks groups heads_per_group head_dim state_dim,"
            " groups heads_per_group chunks chunk_size"
            " -> chunks chunk_size groups heads_per_group head_dim",
        )

        y = y_diag + y_off
        if d is not None:
            d_grouped = rearrange(d, "(groups heads_per_group) -> groups heads_per_group", groups=self.num_groups)
            y = y + d_grouped[None, None, :, :, None] * values
        y = rearrange(
            y,
            "chunks chunk_size groups heads_per_group head_dim"
            " -> (chunks chunk_size) (groups heads_per_group) head_dim",
        )

        if z is not None:
            gate = z + z_bias[None, :, :] if z_bias is not None else z
            y = y * jax.nn.silu(gate)

        y = y[:seq_len]

        new_states_flat = rearrange(
            new_states,
            "chunks groups heads_per_group head_dim state_dim -> chunks (groups heads_per_group) head_dim state_dim",
        )
        final_state = self._compute_final_state(
            values_orig,
            keys_orig,
            dt_orig,
            new_states_flat,
            num_steps,
            chunk_size,
        )

        return y, final_state

    def _compute_final_state(
        self,
        values: Float[Array, "suffix_tokens heads head_dim"],
        keys: Float[Array, "suffix_tokens groups state_dim"],
        dt: Float[Array, "suffix_tokens heads"],
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
            values_chunk = jax.lax.dynamic_slice(
                values,
                (chunk_start_pos, 0, 0),
                (chunk_size, values.shape[1], values.shape[2]),
            )
            keys_chunk = jax.lax.dynamic_slice(
                keys,
                (chunk_start_pos, 0, 0),
                (chunk_size, keys.shape[1], keys.shape[2]),
            )
            dt_chunk = jax.lax.dynamic_slice(dt, (chunk_start_pos, 0), (chunk_size, dt.shape[1]))

            log_decay_cumsum = jnp.cumsum(-dt_chunk, axis=0)
            last_pos_idx = pos_in_chunk - 1
            log_decay_cumsum_at_last = jax.lax.dynamic_index_in_dim(
                log_decay_cumsum,
                last_pos_idx,
                axis=0,
                keepdims=False,
            )

            decayed_start = jnp.exp(log_decay_cumsum_at_last)[:, None, None] * chunk_start_state

            decay_to_last = jnp.exp(log_decay_cumsum_at_last[None, :] - log_decay_cumsum)
            mask = jnp.arange(chunk_size) <= last_pos_idx
            masked_decay = jnp.where(mask[:, None], decay_to_last, 0.0)
            keys_expanded = jnp.repeat(keys_chunk, heads_per_group, axis=1)
            input_contrib = einsum(
                masked_decay,
                keys_expanded,
                values_chunk,
                "chunk_size heads, chunk_size heads state_dim, chunk_size heads head_dim -> heads head_dim state_dim",
            )

            return decayed_start + input_contrib

        return jax.lax.cond(pos_in_chunk == 0, at_boundary, within_chunk)

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: SSMStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> Mamba2Result:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for Mamba2.")

        if state is None:
            state = SSMStateLayer.init(
                self.config.kernel_size,
                self.config.conv_dim,
                (self.num_heads, self.head_dim, self.state_dim),
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

        values = rearrange(
            x_channels,
            "suffix_tokens (heads head_channels) -> suffix_tokens heads head_channels",
            heads=self.num_heads,
        )
        keys = rearrange(
            input_proj_channels,
            "suffix_tokens (groups state_channels) -> suffix_tokens groups state_channels",
            groups=self.num_groups,
        )
        queries = rearrange(
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
            values,
            keys,
            queries,
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
            updated_state = SSMStateLayer(updated_conv_state, final_ssm_state)
        else:
            updated_state = None

        return Mamba2Result(
            outputs=outputs,
            state=updated_state,
        )

    def init_static_state(self, capacity: int) -> SSMStateLayer:  # noqa: ARG002
        return SSMStateLayer.init(
            self.config.kernel_size,
            self.config.conv_dim,
            (self.num_heads, self.head_dim, self.state_dim),
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
