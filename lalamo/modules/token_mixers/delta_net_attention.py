from dataclasses import dataclass
from functools import partial

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_mapping, require_tree
from lalamo.modules.common import Initializer, PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixers.state.ssm_state import SSMStateLayer
from lalamo.modules.utils import vmap_with_key

from .common import MixerForwardPassConfig, TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .convolutions import SeparableCausalConv, SeparableCausalConvConfig

__all__ = [
    "DeltaNetAttention",
    "DeltaNetAttentionConfig",
    "DeltaNetAttentionResult",
]


DeltaNetAttentionResult = TokenMixerResult[SSMStateLayer]


def _delta_dims(
    num_heads: int,
    num_groups: int,
    head_dim: int,
    value_head_dim: int,
) -> tuple[int, int, int]:
    key_dim = num_groups * head_dim
    value_dim = num_heads * value_head_dim
    return key_dim, value_dim, key_dim * 2 + value_dim


@dataclass(frozen=True)
class DeltaNetAttentionConfig(TokenMixerConfigBase):
    in_proj_config: LinearConfig
    conv_config: SeparableCausalConvConfig
    out_proj_config: LinearConfig
    norm_config: NormalizationConfig

    num_heads: int
    num_groups: int
    head_dim: int
    value_head_dim: int
    kernel_size: int

    @property
    def rope_dim(self) -> None:
        return None

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "DeltaNetAttention":
        key_dim, value_dim, conv_dim = _delta_dims(
            self.num_heads,
            self.num_groups,
            self.head_dim,
            self.value_head_dim,
        )
        in_proj = self.in_proj_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(
                key_dim,
                key_dim,
                value_dim,
                value_dim,
                self.num_heads,
                self.num_heads,
            ),
            has_biases=False,
        )
        conv = self.conv_config.init(initializer, conv_dim, self.kernel_size)
        out_proj = self.out_proj_config.init(
            initializer,
            input_dim=value_dim,
            output_dims=(model_dim,),
            has_biases=False,
        )
        norm = self.norm_config.init(initializer, self.value_head_dim)
        dt_bias = initializer.zeros((self.num_heads,), initializer.precision)
        a_log = initializer.zeros((self.num_heads,), initializer.precision)
        return DeltaNetAttention(
            config=self,
            in_proj=in_proj,
            conv=conv,
            out_proj=out_proj,
            norm=norm,
            dt_bias=dt_bias,
            a_log=a_log,
        )


class DeltaNetAttention(TokenMixerBase[DeltaNetAttentionConfig, SSMStateLayer]):
    in_proj: LinearBase
    conv: SeparableCausalConv
    out_proj: Linear
    norm: Normalization
    dt_bias: Float[Array, " heads"]
    a_log: Float[Array, " heads"]

    @property
    def num_heads(self) -> int:
        return self.config.num_heads

    @property
    def num_groups(self) -> int:
        return self.config.num_groups

    @property
    def head_dim(self) -> int:
        return self.config.head_dim

    @property
    def value_head_dim(self) -> int:
        return self.config.value_head_dim

    @property
    def kernel_size(self) -> int:
        return self.config.kernel_size

    @property
    def model_dim(self) -> int:
        return self.in_proj.input_dim

    @property
    def key_dim(self) -> int:
        return self.num_groups * self.head_dim

    @property
    def value_dim(self) -> int:
        return self.num_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    def _recurrent_scan(
        self,
        queries: Float[Array, "tokens heads key_channels"],
        keys: Float[Array, "tokens heads key_channels"],
        values: Float[Array, "tokens heads value_channels"],
        decay_factor: Float[Array, "tokens heads"],
        beta: Float[Array, "tokens heads"],
        initial_state: Float[Array, "heads value_channels key_channels"],
        num_steps: Int[Array, ""] | int,
    ) -> DeltaNetScanResult:
        def scan_fn(
            index_and_state: tuple[Int[Array, ""], Float[Array, "heads value_channels key_channels"]],
            step_inputs: tuple[
                Float[Array, "heads key_channels"],
                Float[Array, "heads key_channels"],
                Float[Array, "heads value_channels"],
                Float[Array, " heads"],
                Float[Array, " heads"],
            ],
        ) -> tuple[
            tuple[Int[Array, ""], Float[Array, "heads value_channels key_channels"]],
            Float[Array, "heads value_channels"],
        ]:
            index, carry_state = index_and_state
            query_t, key_t, value_t, decay_factor_t, beta_t = step_inputs

            decay = jnp.exp(decay_factor_t)[:, None, None]
            decayed_state = carry_state * decay
            value_delta = value_t - jnp.sum(decayed_state * key_t[:, None, :], axis=-1)
            value_delta = value_delta * beta_t[:, None]
            updated_state = decayed_state + value_delta[:, :, None] * key_t[:, None, :]
            output_t = einops.einsum(
                query_t, updated_state, "heads key_channels, heads value_channels key_channels -> heads value_channels"
            )

            propagated_state = jax.lax.cond(index < num_steps, lambda: updated_state, lambda: carry_state)
            return (index + 1, propagated_state), output_t

        (_, final_state), outputs = jax.lax.scan(
            scan_fn,
            (jnp.zeros((), dtype=jnp.int32), initial_state),
            (queries, keys, values, decay_factor, beta),
        )
        return DeltaNetScanResult(outputs, final_state)

    def _chunked_scan(
        self,
        queries: Float[Array, "tokens heads key_channels"],
        keys: Float[Array, "tokens heads key_channels"],
        values: Float[Array, "tokens heads value_channels"],
        decay_factor: Float[Array, "tokens heads"],
        beta: Float[Array, "tokens heads"],
        initial_state: Float[Array, "heads value_channels key_channels"],
        num_steps: Int[Array, ""] | int,
        forward_pass_config: MixerForwardPassConfig,
    ) -> DeltaNetScanResult:
        chunk_size = forward_pass_config.chunk_size
        min_chunk_len = forward_pass_config.min_chunk_len
        num_tokens, _, _ = queries.shape
        num_steps_arr = jnp.asarray(num_steps, dtype=jnp.int32)
        dtype = queries.dtype

        remainder = num_tokens % chunk_size
        has_short_tail = 0 < remainder < min_chunk_len
        num_chunked_tokens = num_tokens - remainder if has_short_tail else num_tokens

        if has_short_tail:
            tail_queries = queries[num_chunked_tokens:]
            tail_keys = keys[num_chunked_tokens:]
            tail_values = values[num_chunked_tokens:]
            tail_decay = decay_factor[num_chunked_tokens:]
            tail_beta = beta[num_chunked_tokens:]
            queries = queries[:num_chunked_tokens]
            keys = keys[:num_chunked_tokens]
            values = values[:num_chunked_tokens]
            decay_factor = decay_factor[:num_chunked_tokens]
            beta = beta[:num_chunked_tokens]

        pad_len = (chunk_size - num_chunked_tokens % chunk_size) % chunk_size if num_chunked_tokens > 0 else 0
        if pad_len > 0:
            queries = jnp.pad(queries, ((0, pad_len), (0, 0), (0, 0)))
            keys = jnp.pad(keys, ((0, pad_len), (0, 0), (0, 0)))
            values = jnp.pad(values, ((0, pad_len), (0, 0), (0, 0)))
            decay_factor = jnp.pad(decay_factor, ((0, pad_len), (0, 0)))
            beta = jnp.pad(beta, ((0, pad_len), (0, 0)))

        padded_len, _, _ = queries.shape
        valid_mask = (jnp.arange(padded_len) < num_steps_arr).astype(dtype)
        keys = keys * valid_mask[:, None, None]
        values = values * valid_mask[:, None, None]
        beta = beta * valid_mask[:, None]
        decay_factor = decay_factor * valid_mask[:, None]

        num_chunks = padded_len // chunk_size
        queries_c = queries.reshape(num_chunks, chunk_size, self.num_heads, self.head_dim)
        keys_c = keys.reshape(num_chunks, chunk_size, self.num_heads, self.head_dim)
        values_c = values.reshape(num_chunks, chunk_size, self.num_heads, self.value_head_dim)
        decay_c = decay_factor.reshape(num_chunks, chunk_size, self.num_heads)
        beta_c = beta.reshape(num_chunks, chunk_size, self.num_heads)

        # Phase 1: intra-chunk scan (parallel across chunks via vmap)
        def _intra_chunk_token_step(
            carry: tuple[
                Float[Array, "heads value_channels key_channels"], Float[Array, "heads key_channels key_channels"]
            ],
            token_inputs: DeltaNetScanInputs,
        ) -> tuple[
            tuple[Float[Array, "heads value_channels key_channels"], Float[Array, "heads key_channels key_channels"]],
            DeltaNetTokenStepOutput,
        ]:
            state, prop = carry

            decay = jnp.exp(token_inputs.decay_factor)[:, None, None]

            decayed_state = state * decay
            state_dot_key = jnp.sum(decayed_state * token_inputs.keys[:, None, :], axis=-1)
            value_delta = (token_inputs.values - state_dot_key) * token_inputs.beta[:, None]
            new_state = decayed_state + value_delta[:, :, None] * token_inputs.keys[:, None, :]

            decayed_prop = prop * decay
            prop_dot_key = einops.einsum(
                decayed_prop,
                token_inputs.keys,
                "heads key_channels_out key_channels_in, heads key_channels_in -> heads key_channels_out",
            )
            new_prop = (
                decayed_prop
                - token_inputs.beta[:, None, None] * prop_dot_key[:, :, None] * token_inputs.keys[:, None, :]
            )

            local_output = einops.einsum(
                token_inputs.queries,
                new_state,
                "heads key_channels, heads value_channels key_channels -> heads value_channels",
            )
            correction_vec = einops.einsum(
                new_prop,
                token_inputs.queries,
                "heads key_channels_out key_channels_in, heads key_channels_in -> heads key_channels_out",
            )

            return (new_state, new_prop), DeltaNetTokenStepOutput(local_output, correction_vec)

        def _intra_chunk_scan(chunk_inputs: DeltaNetScanInputs) -> DeltaNetChunkScanResult:
            state_init = jnp.zeros((self.num_heads, self.value_head_dim, self.head_dim), dtype=dtype)
            prop_init = jnp.tile(jnp.eye(self.head_dim, dtype=dtype), (self.num_heads, 1, 1))
            (end_state, end_prop), step_outputs = jax.lax.scan(
                _intra_chunk_token_step,
                (state_init, prop_init),
                chunk_inputs,
            )
            return DeltaNetChunkScanResult(step_outputs.local_output, step_outputs.correction_vec, end_state, end_prop)

        chunk_results = jax.vmap(_intra_chunk_scan)(
            DeltaNetScanInputs(queries_c, keys_c, values_c, decay_c, beta_c),
        )

        # Phase 2: inter-chunk state propagation (sequential over num_chunks)
        def _inter_chunk_step(
            actual_state: Float[Array, "heads value_channels key_channels"],
            chunk_data: tuple[
                Float[Array, "heads value_channels key_channels"], Float[Array, "heads key_channels key_channels"]
            ],
        ) -> tuple[
            Float[Array, "heads value_channels key_channels"], Float[Array, "heads value_channels key_channels"]
        ]:
            local_end, prop_matrix = chunk_data
            next_state = local_end + einops.einsum(
                actual_state,
                prop_matrix,
                "heads value_channels key_channels_in,"
                " heads key_channels_in key_channels_out"
                " -> heads value_channels key_channels_out",
            )
            return next_state, actual_state

        final_state, chunk_initials = jax.lax.scan(
            _inter_chunk_step,
            initial_state,
            (chunk_results.end_state, chunk_results.end_prop),
        )

        # Phase 3: apply corrections (parallel batched matmul)
        corrections = einops.einsum(
            chunk_initials,
            chunk_results.correction_vecs,
            "chunk heads value_channels key_channels,"
            " chunk time heads key_channels"
            " -> chunk time heads value_channels",
        )
        outputs = chunk_results.chunk_outputs + corrections
        outputs = outputs.reshape(padded_len, self.num_heads, self.value_head_dim)[:num_chunked_tokens]

        if has_short_tail:
            tail_num_steps = jnp.clip(num_steps_arr - num_chunked_tokens, 0, remainder)
            tail_result = self._recurrent_scan(
                tail_queries,
                tail_keys,
                tail_values,
                tail_decay,
                tail_beta,
                final_state,
                tail_num_steps,
            )
            outputs = jnp.concatenate([outputs, tail_result.outputs], axis=0)
            final_state = tail_result.final_state

        return DeltaNetScanResult(outputs, final_state)

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: SSMStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),  # noqa: B008
        *,
        key: PRNGKeyArray | None,
    ) -> DeltaNetAttentionResult:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for DeltaNetAttention.")

        in_key, out_key = jax.random.split(key) if key is not None else (None, None)
        num_tokens, *_ = inputs.shape
        proj_query, proj_key, proj_value, gate, beta_logits, decay_input = vmap_with_key(
            partial(self.in_proj, forward_pass_config=forward_pass_config.arrays),
            inputs,
            key=in_key,
        )
        assert proj_query.shape[0] == num_tokens

        mixed_qkv = jnp.concatenate([proj_query, proj_key, proj_value], axis=-1)
        beta = jax.nn.sigmoid(beta_logits)

        if state is None:
            state = SSMStateLayer.init(
                self.kernel_size,
                self.conv_dim,
                (self.num_heads, self.value_head_dim, self.head_dim),
                self.in_proj.activation_precision,
            )

        conv_output, updated_conv_state = self.conv(
            mixed_qkv,
            length_without_padding,
            state.conv_state,
            return_updated_state=return_updated_state,
        )
        assert conv_output.shape[0] == num_tokens
        conv_output = jax.nn.silu(conv_output)

        query, key, value = jnp.split(conv_output, [self.key_dim, 2 * self.key_dim], axis=-1)

        query = query.reshape(num_tokens, self.num_groups, self.head_dim)
        key = key.reshape(num_tokens, self.num_groups, self.head_dim)
        value = value.reshape(num_tokens, self.num_heads, self.value_head_dim)

        # since we work with exponentials, we (possibly?) uplift dtype to make sure numbers are nice
        decay_factor = -jnp.exp(self.a_log.astype(jnp.float32)) * jax.nn.softplus(
            (decay_input + self.dt_bias).astype(jnp.float32),
        )
        decay_factor = decay_factor.astype(inputs.dtype)

        repeat_factor = self.num_heads // self.num_groups
        if repeat_factor > 1:
            query = jnp.repeat(query, repeat_factor, axis=1)
            key = jnp.repeat(key, repeat_factor, axis=1)

        eps = jnp.array(1e-6, dtype=query.dtype)
        scale = jnp.array(self.head_dim**-0.5, dtype=query.dtype)
        query = query * jax.lax.rsqrt((query * query).sum(axis=-1, keepdims=True) + eps)
        key = key * jax.lax.rsqrt((key * key).sum(axis=-1, keepdims=True) + eps)
        query = query * scale

        if length_without_padding is None:
            length_without_padding = num_tokens

        length_without_padding = jnp.asarray(length_without_padding, dtype=jnp.int32)
        length_without_padding = jnp.clip(length_without_padding, 0, num_tokens)

        if num_tokens < forward_pass_config.min_chunk_len:
            core_attn_out, final_state = self._recurrent_scan(
                query,
                key,
                value,
                decay_factor,
                beta,
                state.ssm_state,
                length_without_padding,
            )
        else:
            core_attn_out, final_state = self._chunked_scan(
                query,
                key,
                value,
                decay_factor,
                beta,
                state.ssm_state,
                length_without_padding,
                forward_pass_config,
            )

        def norm_gate(x: Float[Array, " channels"], gate: Float[Array, " channels"]) -> Float[Array, " channels"]:
            return self.norm(x) * jax.nn.silu(gate)

        num_tokens, *_ = gate.shape
        gate = gate.reshape(num_tokens, self.num_heads, self.value_head_dim)
        core_attn_out = jax.vmap(jax.vmap(norm_gate))(core_attn_out, gate)
        core_attn_out = core_attn_out.reshape(num_tokens, -1)

        (outputs,) = vmap_with_key(
            partial(self.out_proj, forward_pass_config=forward_pass_config.arrays),
            core_attn_out,
            key=out_key,
        )

        if return_updated_state:
            assert updated_conv_state is not None
            updated_state = SSMStateLayer(updated_conv_state, final_state)
        else:
            updated_state = None

        return TokenMixerResult(outputs, updated_state)

    def init_static_state(self, capacity: int) -> SSMStateLayer:  # noqa: ARG002
        return SSMStateLayer.init(
            self.kernel_size,
            self.conv_dim,
            (self.num_heads, self.value_head_dim, self.head_dim),
            self.in_proj.activation_precision,
        )
