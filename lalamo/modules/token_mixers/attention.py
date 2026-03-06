from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from einops import einsum, rearrange
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import dummy_array, require_mapping
from lalamo.modules.common import ParameterTree, PositionalEmbeddingSelector, require_array, require_tree
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.utils import apply_soft_capping

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .state import DynamicKVCacheLayer, KVCacheLayer, StaticKVCacheLayer

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
]


def _repeat_kv(
    keys_or_values: Float[Array, "tokens groups channels"],
    group_size: int,
) -> Float[Array, "tokens groups*group_size channels"]:
    return jnp.repeat(keys_or_values, group_size, axis=1)


def _soft_capped_attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    mask: Bool[Array, "dst_tokens src_tokens"] | None,
    scale: float | None,
    logit_soft_cap: float,
) -> Float[Array, "dst_tokens heads head_channels"]:
    _, num_heads, head_dim = queries.shape
    _, num_groups, _ = keys.shape
    group_size = num_heads // num_groups
    keys = _repeat_kv(keys, group_size)
    values = _repeat_kv(values, group_size)
    queries_head_first = rearrange(queries, "dst_tokens heads channels -> heads dst_tokens channels")
    keys_head_first = rearrange(keys, "src_tokens heads channels -> heads src_tokens channels")
    attention_logits = einsum(
        queries_head_first,
        keys_head_first,
        "heads dst_tokens channels, heads src_tokens channels -> heads dst_tokens src_tokens",
    )
    if mask is not None:
        attention_logits = jnp.where(
            mask,
            attention_logits,
            jnp.array(float("-inf"), dtype=attention_logits.dtype),
        )

    if scale is None:
        scale_val = head_dim**-0.5
    else:
        scale_val = float(scale)
    attention_logits = attention_logits * scale_val
    attention_logits = apply_soft_capping(attention_logits, logit_soft_cap)
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    return einsum(
        attention_weights,
        values,
        "heads dst_tokens src_tokens, src_tokens heads channels -> dst_tokens heads channels",
    )


AttentionResult = TokenMixerResult[KVCacheLayer]


@dataclass(frozen=True)
class AttentionConfig(TokenMixerConfigBase):
    qkv_projection_config: LinearConfig
    out_projection_config: LinearConfig

    query_norm_config: NormalizationConfig | None
    key_norm_config: NormalizationConfig | None

    num_heads: int
    num_groups: int
    head_dim: int
    is_causal: bool
    scale: float | None
    sliding_window_size: int | None

    logit_soft_cap: float | None
    has_sinks: bool
    has_qkv_biases: bool
    has_out_biases: bool
    has_gate: bool = False
    use_rope: bool = True
    # Per-head rotary dimension; if set smaller than head_dim; RoPE is applied to the start of the embedding
    partial_rope_dim: int | None = None

    @property
    def rope_dim(self) -> int | None:
        if not self.use_rope:
            return None
        return self.partial_rope_dim if self.partial_rope_dim is not None else self.head_dim

    def random_init(
        self,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "Attention":
        qkv_key, out_key = jax.random.split(key)
        q_output_dim = self.num_heads * self.head_dim
        output_dims = (
            q_output_dim,
            self.num_groups * self.head_dim,
            self.num_groups * self.head_dim,
        )
        if self.has_gate:
            output_dims = (*output_dims, q_output_dim)
        qkv_projection = self.qkv_projection_config.random_init(
            input_dim=model_dim,
            output_dims=output_dims,
            has_biases=self.has_qkv_biases,
            key=qkv_key,
        )
        out_projection = self.out_projection_config.random_init(
            self.num_heads * self.head_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
            key=out_key,
        )

        if self.query_norm_config is not None:
            query_norm = self.query_norm_config.init(
                input_dim=self.head_dim,
            )
        else:
            query_norm = None

        if self.key_norm_config is not None:
            key_norm = self.key_norm_config.init(
                input_dim=self.head_dim,
            )
        else:
            key_norm = None

        if self.has_sinks:
            sinks = jnp.zeros((self.num_heads,), dtype=qkv_projection.activation_precision)
        else:
            sinks = None

        return Attention(
            self,
            qkv_projection=qkv_projection,
            out_projection=out_projection,
            query_norm=query_norm,
            key_norm=key_norm,
            sinks=sinks,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            is_causal=self.is_causal,
            scale=self.scale,
            sliding_window_size=self.sliding_window_size,
            use_rope=self.use_rope,
        )

    def empty(
        self,
        model_dim: int,
    ) -> "Attention":
        q_output_dim = self.num_heads * self.head_dim
        output_dims = (
            q_output_dim,
            self.num_groups * self.head_dim,
            self.num_groups * self.head_dim,
        )
        if self.has_gate:
            output_dims = (*output_dims, q_output_dim)
        qkv_projection = self.qkv_projection_config.empty(
            input_dim=model_dim,
            output_dims=output_dims,
            has_biases=self.has_qkv_biases,
        )
        out_projection = self.out_projection_config.empty(
            self.num_heads * self.head_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
        )

        if self.query_norm_config is not None:
            query_norm = self.query_norm_config.empty(
                input_dim=self.head_dim,
            )
        else:
            query_norm = None

        if self.key_norm_config is not None:
            key_norm = self.key_norm_config.empty(
                input_dim=self.head_dim,
            )
        else:
            key_norm = None

        if self.has_sinks:
            sinks = dummy_array(self.num_heads, qkv_projection.activation_precision)
        else:
            sinks = None

        return Attention(
            self,
            qkv_projection=qkv_projection,
            out_projection=out_projection,
            query_norm=query_norm,
            key_norm=key_norm,
            sinks=sinks,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            is_causal=self.is_causal,
            scale=self.scale,
            sliding_window_size=self.sliding_window_size,
            use_rope=self.use_rope,
        )


class Attention(TokenMixerBase[AttentionConfig, KVCacheLayer]):
    qkv_projection: LinearBase
    out_projection: LinearBase

    query_norm: Normalization | None
    key_norm: Normalization | None

    sinks: Float[Array, " heads"] | None

    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    is_causal: bool = eqx.field(static=True)

    scale: float | None = eqx.field(static=True)
    sliding_window_size: int | None = eqx.field(static=True)
    use_rope: bool = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.qkv_projection.activation_precision

    @property
    def model_dim(self) -> int:
        return self.qkv_projection.input_dim

    @property
    def group_size(self) -> int:
        return self.num_heads // self.num_groups

    @property
    def use_sliding_window(self) -> bool:
        return self.sliding_window_size is not None

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        if not self.use_rope:
            return PositionalEmbeddingSelector.NONE
        if self.use_sliding_window:
            return PositionalEmbeddingSelector.LOCAL
        return PositionalEmbeddingSelector.GLOBAL

    @property
    def has_sinks(self) -> bool:
        return self.sinks is not None

    def __post_init__(self) -> None:
        if self.use_rope != self.config.use_rope:
            raise ValueError(
                f"use_rope {self.use_rope} does not match the specified config use_rope {self.config.use_rope}",
            )
        if self.qkv_projection.has_biases != self.config.has_qkv_biases:
            raise ValueError(
                f"QKV projection has_biases {self.qkv_projection.has_biases} does not match"
                f" the specified config has_qkv_biases {self.config.has_qkv_biases}",
            )
        if self.out_projection.has_biases != self.config.has_out_biases:
            raise ValueError(
                f"Output projection has_biases {self.out_projection.has_biases} does not match"
                f" the specified config has_out_biases {self.config.has_out_biases}",
            )
        if self.query_norm is not None and self.query_norm.input_dim != self.head_dim:
            raise ValueError(
                f"Query normalization input dimension must match head_dim ({self.head_dim}),"
                f" got {self.query_norm.input_dim}",
            )
        if self.key_norm is not None and self.key_norm.input_dim != self.head_dim:
            raise ValueError(
                f"Key normalization input dimension must match head_dim ({self.head_dim}),"
                f" got {self.key_norm.input_dim}",
            )
        if self.num_heads % self.num_groups != 0:
            raise ValueError(
                "Number of heads must be divisible by the number of groups,"
                f" got {self.num_heads} heads and {self.num_groups} groups",
            )
        if self.out_projection.input_dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"Output projection input dimension must be num_heads * head_dim"
                f" ({self.num_heads} * {self.head_dim} = {self.num_heads * self.head_dim}),"
                f" got {self.out_projection.input_dim}",
            )
        output_dims = self.qkv_projection.output_dims
        if len(output_dims) not in (3, 4):
            raise ValueError(
                f"QKV projection must have 3 (qkv) or 4 (qkvz) output dims, got {len(output_dims)}",
            )
        q_output_dim, k_output_dim, v_output_dim = output_dims[:3]
        expected_q = self.num_heads * self.head_dim
        if q_output_dim != expected_q:
            raise ValueError(
                f"Query projection output dimension must be {expected_q}, got {q_output_dim}",
            )
        if k_output_dim != self.num_groups * self.head_dim:
            raise ValueError(
                f"Key projection output dimension must be num_groups * head_dim"
                f" ({self.num_groups} * {self.head_dim} = {self.num_groups * self.head_dim}),"
                f" got {k_output_dim}",
            )
        if v_output_dim != self.num_groups * self.head_dim:
            raise ValueError(
                f"Value projection output dimension must be num_groups * head_dim"
                f" ({self.num_groups} * {self.head_dim} = {self.num_groups * self.head_dim}),"
                f" got {v_output_dim}",
            )
        if self.config.has_gate:
            if len(output_dims) != 4:
                raise ValueError("QKVZ projection must have 4 output dims when has_gate is enabled.")
            gate_output_dim = output_dims[3]
            if gate_output_dim != expected_q:
                raise ValueError(
                    f"Gate projection output dimension must be {expected_q}, got {gate_output_dim}",
                )
        elif len(output_dims) != 3:
            raise ValueError("QKV projection must have 3 output dims when has_gate is disabled.")
        if self.sinks is not None:
            (num_sink_heads,) = self.sinks.shape
            if num_sink_heads != self.num_heads:
                raise ValueError(
                    f"Number of sink heads must be equal to number of heads ({self.num_heads}), got {num_sink_heads}",
                )

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: KVCacheLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> AttentionResult:
        if self.config.has_gate:
            queries, keys, values, gate = vmap(self.qkv_projection, in_axes=0)(inputs)
        else:
            queries, keys, values = vmap(self.qkv_projection, in_axes=0)(inputs)
            gate = None

        queries = rearrange(
            queries,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=self.num_heads,
            head_channels=self.head_dim,
        )
        keys = rearrange(
            keys,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.num_groups,
            head_channels=self.head_dim,
        )
        values = rearrange(
            values,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.num_groups,
            head_channels=self.head_dim,
        )

        if self.query_norm is not None:
            queries = vmap(vmap(self.query_norm))(queries)
        if self.key_norm is not None:
            keys = vmap(vmap(self.key_norm))(keys)

        if positional_embeddings is not None:
            apply_positional_embeddings = vmap(positional_embeddings.apply, in_axes=1, out_axes=1)
            queries = apply_positional_embeddings(queries)
            keys = apply_positional_embeddings(keys)

        if state is None:
            updated_state = DynamicKVCacheLayer.init(self.has_sinks, keys, values, length=length_without_padding)
        else:
            updated_state = state.extend(keys, values, added_length=length_without_padding)

        num_suffix_tokens, _, _ = queries.shape
        mask = updated_state.attention_mask(
            num_suffix_tokens,
            self.is_causal,
            length_without_padding,
            self.sliding_window_size,
        )
        if self.sinks is not None:
            sink_bias = jnp.zeros((self.num_heads, *mask.shape), dtype=queries.dtype)
            sink_bias = sink_bias.at[:, :, 0].set(self.sinks[:, None])
        else:
            sink_bias = None

        if self.config.logit_soft_cap is not None:
            attention_output = _soft_capped_attention_kernel(
                queries,
                updated_state.keys,
                updated_state.values,
                mask=mask,
                scale=self.scale,
                logit_soft_cap=self.config.logit_soft_cap,
            )
        else:
            attention_output = jax.nn.dot_product_attention(
                queries,
                updated_state.keys,
                updated_state.values,
                bias=sink_bias,
                mask=mask,
                scale=self.scale,
            )
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.num_heads,
            head_channels=self.head_dim,
        )
        if gate is not None:
            attention_output = attention_output * jax.nn.sigmoid(gate)
        (result,) = vmap(self.out_projection, in_axes=0)(attention_output)

        if not return_updated_state:
            updated_state = None

        return AttentionResult(
            outputs=result,
            state=updated_state,
        )

    def init_static_state(self, capacity: int) -> StaticKVCacheLayer:
        return StaticKVCacheLayer.init(
            self.has_sinks,
            capacity,
            self.num_groups,
            self.head_dim,
            self.activation_precision,
        )

    def export_weights(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = {
            "qkv_projection": self.qkv_projection.export_weights(),
            "out_projection": self.out_projection.export_weights(),
        }
        if self.query_norm is not None:
            result["query_norm"] = self.query_norm.export_weights()
        if self.key_norm is not None:
            result["key_norm"] = self.key_norm.export_weights()
        if self.sinks is not None:
            assert isinstance(self.sinks, Array)
            result["sinks"] = self.sinks
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        return replace(
            self,
            qkv_projection=self.qkv_projection.import_weights(require_tree(weights["qkv_projection"])),
            out_projection=self.out_projection.import_weights(require_tree(weights["out_projection"])),
            query_norm=self.query_norm.import_weights(require_tree(weights["query_norm"]))
            if self.query_norm
            else None,
            key_norm=self.key_norm.import_weights(require_tree(weights["key_norm"])) if self.key_norm else None,
            sinks=require_array(weights["sinks"]) if self.sinks is not None else None,
        )
