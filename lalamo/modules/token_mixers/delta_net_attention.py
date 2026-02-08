from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_array, require_tree
from lalamo.modules.common import PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixers.state.ssm_state import SSMStateLayer

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
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

    def random_init(
        self,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "DeltaNetAttention":
        proj_key, conv_key, out_key = jax.random.split(key, 3)
        key_dim, value_dim, conv_dim = _delta_dims(
            self.num_heads,
            self.num_groups,
            self.head_dim,
            self.value_head_dim,
        )
        in_proj = self.in_proj_config.random_init(
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
            key=proj_key,
        )
        conv = self.conv_config.random_init(conv_dim, self.kernel_size, key=conv_key)
        out_proj = self.out_proj_config.random_init(
            input_dim=value_dim,
            output_dims=(model_dim,),
            has_biases=False,
            key=out_key,
        )
        norm = self.norm_config.init(self.value_head_dim)
        dt_bias = jnp.zeros((self.num_heads,), dtype=in_proj.activation_precision)
        a_log = jnp.zeros((self.num_heads,), dtype=in_proj.activation_precision)
        return DeltaNetAttention(
            self,
            in_proj=in_proj,
            conv=conv,
            out_proj=out_proj,
            norm=norm,
            dt_bias=dt_bias,
            a_log=a_log,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            value_head_dim=self.value_head_dim,
            kernel_size=self.kernel_size,
        )

    def empty(
        self,
        model_dim: int,
    ) -> "DeltaNetAttention":
        key_dim, value_dim, conv_dim = _delta_dims(
            self.num_heads,
            self.num_groups,
            self.head_dim,
            self.value_head_dim,
        )
        in_proj = self.in_proj_config.empty(
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
        conv = self.conv_config.empty(conv_dim, self.kernel_size)
        out_proj = self.out_proj_config.empty(
            input_dim=value_dim,
            output_dims=(model_dim,),
            has_biases=False,
        )
        norm = self.norm_config.empty(self.value_head_dim)
        dt_bias = dummy_array((self.num_heads,), dtype=in_proj.activation_precision)
        a_log = dummy_array((self.num_heads,), dtype=in_proj.activation_precision)
        return DeltaNetAttention(
            self,
            in_proj=in_proj,
            conv=conv,
            out_proj=out_proj,
            norm=norm,
            dt_bias=dt_bias,
            a_log=a_log,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            value_head_dim=self.value_head_dim,
            kernel_size=self.kernel_size,
        )


class DeltaNetAttention(TokenMixerBase[DeltaNetAttentionConfig, SSMStateLayer]):
    in_proj: LinearBase
    conv: SeparableCausalConv
    out_proj: LinearBase
    norm: Normalization
    dt_bias: Float[Array, " heads"]
    a_log: Float[Array, " heads"]
    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    value_head_dim: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.in_proj.activation_precision

    @property
    def model_dim(self) -> int:
        return self.in_proj.input_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return PositionalEmbeddingSelector.NONE

    @property
    def key_dim(self) -> int:
        return self.num_groups * self.head_dim

    @property
    def value_dim(self) -> int:
        return self.num_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    def _scan(
        self,
        queries: Float[Array, "tokens heads head_k_dim"],
        keys: Float[Array, "tokens heads head_k_dim"],
        values: Float[Array, "tokens heads head_v_dim"],
        decay_factor: Float[Array, "tokens heads"],
        beta: Float[Array, "tokens heads"],
        initial_state: Float[Array, "heads head_k_dim head_v_dim"],
        num_steps: Int[Array, ""] | int,
    ) -> tuple[
        Float[Array, "tokens heads head_v_dim"],
        Float[Array, "heads head_k_dim head_v_dim"],
    ]:
        def scan_fn(
            index_and_state: tuple[Int[Array, ""], Float[Array, "heads head_k_dim head_v_dim"]],
            step_inputs: tuple[
                Float[Array, "heads head_k_dim"],
                Float[Array, "heads head_k_dim"],
                Float[Array, "heads head_v_dim"],
                Float[Array, " heads"],
                Float[Array, " heads"],
            ],
        ) -> tuple[
            tuple[Int[Array, ""], Float[Array, "heads head_k_dim head_v_dim"]],
            Float[Array, "heads head_v_dim"],
        ]:
            index, carry_state = index_and_state
            query_t, key_t, value_t, decay_factor_t, beta_t = step_inputs

            decay = jnp.exp(decay_factor_t)[:, None, None]
            decayed_state = carry_state * decay
            value_delta = value_t - jnp.sum(decayed_state * key_t[:, :, None], axis=-2)
            value_delta = value_delta * beta_t[:, None]
            updated_state = decayed_state + key_t[:, :, None] * value_delta[:, None, :]
            output_t = jnp.einsum("hk,hkv->hv", query_t, updated_state)

            propagated_state = jax.lax.cond(index < num_steps, lambda: updated_state, lambda: carry_state)
            return (index + 1, propagated_state), output_t

        (_, final_state), outputs = jax.lax.scan(
            scan_fn,
            (jnp.zeros((), dtype=jnp.int32), initial_state),
            (queries, keys, values, decay_factor, beta),
        )
        return outputs, final_state

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: SSMStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> DeltaNetAttentionResult:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for DeltaNetAttention.")

        num_tokens, *_ = inputs.shape
        proj_query, proj_key, proj_value, gate, beta_logits, decay_input = vmap(self.in_proj)(inputs)
        assert proj_query.shape[0] == num_tokens

        mixed_qkv = jnp.concatenate([proj_query, proj_key, proj_value], axis=-1)
        beta = jax.nn.sigmoid(beta_logits)

        if state is None:
            state = SSMStateLayer.init(
                self.kernel_size,
                self.conv_dim,
                (self.num_heads, self.head_dim, self.value_head_dim),
                self.activation_precision,
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

        core_attn_out, final_state = self._scan(
            query,
            key,
            value,
            decay_factor,
            beta,
            state.ssm_state,
            length_without_padding,
        )

        def norm_gate(x: Float[Array, " channels"], gate: Float[Array, " channels"]) -> Float[Array, " channels"]:
            return self.norm(x) * jax.nn.silu(gate.astype(jnp.float32)).astype(x.dtype)

        num_tokens, *_ = gate.shape
        gate = gate.reshape(num_tokens, self.num_heads, self.value_head_dim)
        core_attn_out = jax.vmap(jax.vmap(norm_gate))(core_attn_out, gate)
        core_attn_out = core_attn_out.reshape(num_tokens, -1)

        (outputs,) = vmap(self.out_proj)(core_attn_out)

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
            (self.num_heads, self.head_dim, self.value_head_dim),
            self.activation_precision,
        )

    def export_weights(self) -> ParameterTree:
        return {
            "in_proj": self.in_proj.export_weights(),
            "conv": self.conv.export_weights(),
            "out_proj": self.out_proj.export_weights(),
            "norm": self.norm.export_weights(),
            "dt_bias": self.dt_bias,
            "a_log": self.a_log,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            in_proj=self.in_proj.import_weights(require_tree(weights["in_proj"])),
            conv=self.conv.import_weights(require_tree(weights["conv"])),
            out_proj=self.out_proj.import_weights(require_tree(weights["out_proj"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
            dt_bias=require_array(weights["dt_bias"]),
            a_log=require_array(weights["a_log"]),
        )
