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

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .mamba import SeparableCausalConv, SeparableCausalConvConfig
from .state import DeltaNetStateLayer

__all__ = [
    "DeltaNetAttention",
    "DeltaNetAttentionConfig",
    "DeltaNetAttentionResult",
]


DeltaNetAttentionResult = TokenMixerResult[DeltaNetStateLayer]


@dataclass(frozen=True)
class DeltaNetAttentionConfig(TokenMixerConfigBase):
    in_proj_qkvz_config: LinearConfig
    in_proj_ba_config: LinearConfig
    conv_config: SeparableCausalConvConfig
    out_proj_config: LinearConfig
    norm_config: NormalizationConfig

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    kernel_size: int

    @property
    def rope_dim(self) -> None:
        return None

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    def random_init(
        self,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "DeltaNetAttention":
        qkvz_key, ba_key, conv_key, out_key = jax.random.split(key, 4)
        in_proj_qkvz = self.in_proj_qkvz_config.random_init(
            input_dim=model_dim,
            output_dims=(self.key_dim * 2 + self.value_dim * 2,),
            has_biases=False,
            key=qkvz_key,
        )
        in_proj_ba = self.in_proj_ba_config.random_init(
            input_dim=model_dim,
            output_dims=(self.num_v_heads * 2,),
            has_biases=False,
            key=ba_key,
        )
        conv = self.conv_config.random_init(self.conv_dim, self.kernel_size, key=conv_key)
        out_proj = self.out_proj_config.random_init(
            input_dim=self.value_dim,
            output_dims=(model_dim,),
            has_biases=False,
            key=out_key,
        )
        norm = self.norm_config.init(self.head_v_dim)
        dt_bias = jnp.zeros((self.num_v_heads,), dtype=in_proj_qkvz.activation_precision)
        a_log = jnp.zeros((self.num_v_heads,), dtype=in_proj_qkvz.activation_precision)
        return DeltaNetAttention(
            self,
            in_proj_qkvz=in_proj_qkvz,
            in_proj_ba=in_proj_ba,
            conv=conv,
            out_proj=out_proj,
            norm=norm,
            dt_bias=dt_bias,
            a_log=a_log,
        )

    def empty(
        self,
        model_dim: int,
    ) -> "DeltaNetAttention":
        in_proj_qkvz = self.in_proj_qkvz_config.empty(
            input_dim=model_dim,
            output_dims=(self.key_dim * 2 + self.value_dim * 2,),
            has_biases=False,
        )
        in_proj_ba = self.in_proj_ba_config.empty(
            input_dim=model_dim,
            output_dims=(self.num_v_heads * 2,),
            has_biases=False,
        )
        conv = self.conv_config.empty(self.conv_dim, self.kernel_size)
        out_proj = self.out_proj_config.empty(
            input_dim=self.value_dim,
            output_dims=(model_dim,),
            has_biases=False,
        )
        norm = self.norm_config.empty(self.head_v_dim)
        dt_bias = dummy_array((self.num_v_heads,), dtype=in_proj_qkvz.activation_precision)
        a_log = dummy_array((self.num_v_heads,), dtype=in_proj_qkvz.activation_precision)
        return DeltaNetAttention(
            self,
            in_proj_qkvz=in_proj_qkvz,
            in_proj_ba=in_proj_ba,
            conv=conv,
            out_proj=out_proj,
            norm=norm,
            dt_bias=dt_bias,
            a_log=a_log,
        )


class DeltaNetAttention(TokenMixerBase[DeltaNetAttentionConfig, DeltaNetStateLayer]):
    in_proj_qkvz: LinearBase
    in_proj_ba: LinearBase
    conv: SeparableCausalConv
    out_proj: LinearBase
    norm: Normalization
    dt_bias: Float[Array, " heads"]
    a_log: Float[Array, " heads"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.in_proj_qkvz.activation_precision

    @property
    def model_dim(self) -> int:
        return self.in_proj_qkvz.input_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return PositionalEmbeddingSelector.NONE

    def _split_qkvz_ba(
        self,
        mixed_qkvz: Float[Array, "tokens channels"],
        mixed_ba: Float[Array, "tokens channels"],
    ) -> tuple[
        Float[Array, "tokens k_heads k_dim"],
        Float[Array, "tokens k_heads k_dim"],
        Float[Array, "tokens v_heads v_dim"],
        Float[Array, "tokens v_heads v_dim"],
        Float[Array, "tokens v_heads"],
        Float[Array, "tokens v_heads"],
    ]:
        num_k_heads = self.config.num_k_heads
        num_v_heads = self.config.num_v_heads
        if num_v_heads % num_k_heads != 0:
            raise ValueError(
                "Number of value heads must be divisible by number of key heads, "
                f"got {num_v_heads} and {num_k_heads}.",
            )
        v_per_k = num_v_heads // num_k_heads
        head_k_dim = self.config.head_k_dim
        head_v_dim = self.config.head_v_dim

        mixed_qkvz = mixed_qkvz.reshape(
            mixed_qkvz.shape[0],
            num_k_heads,
            2 * head_k_dim + 2 * head_v_dim * v_per_k,
        )
        mixed_ba = mixed_ba.reshape(mixed_ba.shape[0], num_k_heads, 2 * v_per_k)

        split_points = (
            head_k_dim,
            2 * head_k_dim,
            2 * head_k_dim + head_v_dim * v_per_k,
        )
        query, key, value, z = jnp.split(mixed_qkvz, split_points, axis=-1)
        b, a = jnp.split(mixed_ba, 2, axis=-1)

        value = value.reshape(value.shape[0], num_k_heads, v_per_k, head_v_dim)
        z = z.reshape(z.shape[0], num_k_heads, v_per_k, head_v_dim)
        value = value.reshape(value.shape[0], num_v_heads, head_v_dim)
        z = z.reshape(z.shape[0], num_v_heads, head_v_dim)
        b = b.reshape(b.shape[0], num_v_heads)
        a = a.reshape(a.shape[0], num_v_heads)
        return query, key, value, z, b, a

    def _scan(
        self,
        queries: Float[Array, "tokens heads head_k_dim"],
        keys: Float[Array, "tokens heads head_k_dim"],
        values: Float[Array, "tokens heads head_v_dim"],
        g: Float[Array, "tokens heads"],
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
            query_t, key_t, value_t, g_t, beta_t = step_inputs

            decay = jnp.exp(g_t)[:, None, None]
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
            (queries, keys, values, g, beta),
        )
        return outputs, final_state

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: DeltaNetStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,  # noqa: ARG002
    ) -> DeltaNetAttentionResult:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for DeltaNetAttention.")

        (mixed_qkvz,) = vmap(self.in_proj_qkvz)(inputs)
        (mixed_ba,) = vmap(self.in_proj_ba)(inputs)
        query, key, value, z, b, a = self._split_qkvz_ba(mixed_qkvz, mixed_ba)

        query = query.reshape(query.shape[0], -1)
        key = key.reshape(key.shape[0], -1)
        value = value.reshape(value.shape[0], -1)
        mixed_qkv = jnp.concatenate([query, key, value], axis=-1)

        if state is None:
            state = DeltaNetStateLayer.init(
                self.config.kernel_size,
                self.config.conv_dim,
                self.config.num_v_heads,
                self.config.head_k_dim,
                self.config.head_v_dim,
                self.activation_precision,
            )

        conv_output, updated_conv_state = self.conv(
            mixed_qkv,
            length_without_padding,
            state.conv_state,
            return_updated_state=return_updated_state,
        )
        conv_output = jax.nn.silu(conv_output)

        key_dim = self.config.key_dim
        value_dim = self.config.value_dim
        query, key, value = jnp.split(conv_output, [key_dim, 2 * key_dim], axis=-1)
        query = query.reshape(query.shape[0], self.config.num_k_heads, self.config.head_k_dim)
        key = key.reshape(key.shape[0], self.config.num_k_heads, self.config.head_k_dim)
        value = value.reshape(value.shape[0], self.config.num_v_heads, self.config.head_v_dim)

        beta = jax.nn.sigmoid(b)
        g = -jnp.exp(self.a_log.astype(jnp.float32)) * jax.nn.softplus(
            (a + self.dt_bias).astype(jnp.float32),
        )
        g = g.astype(inputs.dtype)

        repeat_factor = self.config.num_v_heads // self.config.num_k_heads
        if repeat_factor > 1:
            query = jnp.repeat(query, repeat_factor, axis=1)
            key = jnp.repeat(key, repeat_factor, axis=1)

        eps = jnp.array(1e-6, dtype=query.dtype)
        scale = jnp.array(self.config.head_k_dim**-0.5, dtype=query.dtype)
        query = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + eps)
        key = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + eps)
        query = query * scale

        if length_without_padding is None:
            length_without_padding = inputs.shape[0]
        length_without_padding = jnp.asarray(length_without_padding, dtype=jnp.int32)
        length_without_padding = jnp.clip(length_without_padding, 0, inputs.shape[0])

        core_attn_out, final_state = self._scan(
            query,
            key,
            value,
            g,
            beta,
            state.recurrent_state,
            length_without_padding,
        )

        def norm_gate(x: Float[Array, " channels"], gate: Float[Array, " channels"]) -> Float[Array, " channels"]:
            input_dtype = x.dtype
            x_up = x.astype(self.norm.config.accumulation_precision)
            variance = jnp.mean(jnp.square(x_up), axis=-1, keepdims=True)
            x_norm = x_up * jax.lax.rsqrt(variance + self.norm.config.epsilon)
            x_norm = x_norm.astype(input_dtype)
            scaled = (x_norm * self.norm.scales.astype(input_dtype)).astype(jnp.float32)
            gated = scaled * jax.nn.silu(gate.astype(jnp.float32))
            return gated.astype(input_dtype)

        core_attn_out = jax.vmap(jax.vmap(norm_gate))(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], -1)

        (outputs,) = vmap(self.out_proj)(core_attn_out)

        if return_updated_state:
            assert updated_conv_state is not None
            updated_state = DeltaNetStateLayer(updated_conv_state, final_state)
        else:
            updated_state = None

        return TokenMixerResult(outputs, updated_state)

    def init_static_state(self, capacity: int) -> DeltaNetStateLayer:  # noqa: ARG002
        return DeltaNetStateLayer.init(
            self.config.kernel_size,
            self.config.conv_dim,
            self.config.num_v_heads,
            self.config.head_k_dim,
            self.config.head_v_dim,
            self.activation_precision,
        )

    def export_weights(self) -> ParameterTree:
        return {
            "in_proj_qkvz": self.in_proj_qkvz.export_weights(),
            "in_proj_ba": self.in_proj_ba.export_weights(),
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
            in_proj_qkvz=self.in_proj_qkvz.import_weights(require_tree(weights["in_proj_qkvz"])),
            in_proj_ba=self.in_proj_ba.import_weights(require_tree(weights["in_proj_ba"])),
            conv=self.conv.import_weights(require_tree(weights["conv"])),
            out_proj=self.out_proj.import_weights(require_tree(weights["out_proj"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
            dt_bias=require_array(weights["dt_bias"]),
            a_log=require_array(weights["a_log"]),
        )
