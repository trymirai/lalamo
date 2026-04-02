from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from einops import einsum, rearrange
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.common import require_mapping
from lalamo.modules.common import Initializer, ParameterTree, PositionalEmbeddingSelector, require_array, require_tree
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
    gate_projection_config: LinearConfig | None = None
    use_rope: bool = True
    # Per-head rotary dimension; if set smaller than head_dim; RoPE is applied to the start of the embedding
    partial_rope_dim: int | None = None

    @property
    def rope_dim(self) -> int | None:
        if not self.use_rope:
            return None
        return self.partial_rope_dim if self.partial_rope_dim is not None else self.head_dim

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "Attention":
        q_output_dim = self.num_heads * self.head_dim
        output_dims = (
            q_output_dim,
            self.num_groups * self.head_dim,
            self.num_groups * self.head_dim,
        )
        qkv_projection = self.qkv_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=output_dims,
            has_biases=self.has_qkv_biases,
        )
        out_projection = self.out_projection_config.init(
            initializer,
            self.num_heads * self.head_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
        )

        if self.gate_projection_config is not None:
            gate_projection = self.gate_projection_config.init(
                initializer,
                input_dim=model_dim,
                output_dims=(q_output_dim,),
                has_biases=False,
            )
        else:
            gate_projection = None

        if self.query_norm_config is not None:
            query_norm = self.query_norm_config.init(
                initializer,
                input_dim=self.head_dim,
            )
        else:
            query_norm = None

        if self.key_norm_config is not None:
            key_norm = self.key_norm_config.init(
                initializer,
                input_dim=self.head_dim,
            )
        else:
            key_norm = None

        if self.has_sinks:
            sinks = initializer.zeros((self.num_heads,), qkv_projection.activation_precision)
        else:
            sinks = None

        return Attention(
            config=self,
            qkv_projection=qkv_projection,
            gate_projection=gate_projection,
            out_projection=out_projection,
            query_norm=query_norm,
            key_norm=key_norm,
            sinks=sinks,
        )


class Attention(TokenMixerBase[AttentionConfig, KVCacheLayer]):
    qkv_projection: LinearBase
    gate_projection: LinearBase | None
    out_projection: LinearBase

    query_norm: Normalization | None
    key_norm: Normalization | None

    sinks: Float[Array, " heads"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.qkv_projection.activation_precision

    @property
    def model_dim(self) -> int:
        return self.qkv_projection.input_dim

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
    def is_causal(self) -> bool:
        return self.config.is_causal

    @property
    def scale(self) -> float | None:
        return self.config.scale

    @property
    def sliding_window_size(self) -> int | None:
        return self.config.sliding_window_size

    @property
    def logit_soft_cap(self) -> float | None:
        return self.config.logit_soft_cap

    @property
    def use_rope(self) -> bool:
        return self.config.use_rope

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

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: KVCacheLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> AttentionResult:
        queries, keys, values = vmap(self.qkv_projection, in_axes=0)(inputs)
        if self.gate_projection is not None:
            (gate,) = vmap(self.gate_projection, in_axes=0)(inputs)
        else:
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

        if self.logit_soft_cap is not None:
            attention_output = _soft_capped_attention_kernel(
                queries,
                updated_state.keys,
                updated_state.values,
                mask=mask,
                scale=self.scale,
                logit_soft_cap=self.logit_soft_cap,
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
        if self.gate_projection is not None:
            result["gate_projection"] = self.gate_projection.export_weights()
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
            gate_projection=self.gate_projection.import_weights(require_tree(weights["gate_projection"]))
            if self.gate_projection
            else None,
            out_projection=self.out_projection.import_weights(require_tree(weights["out_projection"])),
            query_norm=self.query_norm.import_weights(require_tree(weights["query_norm"]))
            if self.query_norm
            else None,
            key_norm=self.key_norm.import_weights(require_tree(weights["key_norm"])) if self.key_norm else None,
            sinks=require_array(weights["sinks"]) if self.sinks is not None else None,
        )
