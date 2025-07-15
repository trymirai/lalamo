from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from einops import einsum, rearrange, repeat
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterDict
from lalamo.modules.normalization import RMSNorm, RMSNormConfig

from .common import AttentionType, LalamoModule, WeightLayout
from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, StaticKVCacheLayer
from .linear import LinearBase, LinearConfig
from .rope import PositionalEmbeddings
from .utils import apply_soft_capping

__all__ = [
    "Attention",
    "AttentionConfig",
]


def _repeat_kv(
    keys_or_values: Float[Array, "tokens groups channels"],
    group_size: int,
) -> Float[Array, "tokens groups*group_size channels"]:
    return repeat(
        keys_or_values,
        "tokens groups channels -> tokens (groups group_size) channels",
        group_size=group_size,
    )


def _soft_capped_attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    mask: Bool[Array, "dst_tokens src_tokens"] | None,
    scale: float | None,
    logit_soft_cap: float,
) -> Float[Array, "dst_tokens heads head_channels"]:
    dst_length, num_heads, head_dim = queries.shape
    src_length, num_groups, _ = keys.shape
    if scale is None:
        scale = head_dim**-0.5
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
        attention_logits = jnp.where(mask, attention_logits, jnp.array(float("-inf"), dtype=attention_logits.dtype))

    attention_logits = attention_logits * scale
    attention_logits = apply_soft_capping(attention_logits, logit_soft_cap)
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    return einsum(
        attention_weights,
        values,
        "heads dst_tokens src_tokens, src_tokens heads channels -> dst_tokens heads channels",
    )


class AttentionResult(NamedTuple):
    outputs: Float[Array, "suffix_tokens channels"]
    kv_cache: KVCacheLayer | None = None


@dataclass(frozen=True)
class AttentionConfig:
    qkv_projection_config: LinearConfig
    out_projection_config: LinearConfig

    query_norm_config: RMSNormConfig | None
    key_norm_config: RMSNormConfig | None

    logit_soft_cap: float | None
    has_qkv_biases: bool
    has_out_biases: bool

    def random_init(
        self,
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        is_causal: bool,
        scale: float | None,
        sliding_window_size: int | None,
        *,
        key: PRNGKeyArray,
    ) -> "Attention":
        qkv_key, out_key = jax.random.split(key)
        qkv_projection = self.qkv_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(
                num_heads * head_dim,
                num_groups * head_dim,
                num_groups * head_dim,
            ),
            has_biases=self.has_qkv_biases,
            key=qkv_key,
        )
        out_projection = self.out_projection_config.random_init(
            num_heads * head_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
            key=out_key,
        )

        if self.query_norm_config is not None:
            query_norm = self.query_norm_config.init(
                channels=head_dim,
            )
        else:
            query_norm = None

        if self.key_norm_config is not None:
            key_norm = self.key_norm_config.init(
                channels=head_dim,
            )
        else:
            key_norm = None

        return Attention(
            self,
            qkv_projection=qkv_projection,
            out_projection=out_projection,
            query_norm=query_norm,
            key_norm=key_norm,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            is_causal=is_causal,
            scale=scale,
            sliding_window_size=sliding_window_size,
        )


class Attention(LalamoModule[AttentionConfig]):
    qkv_projection: LinearBase
    out_projection: LinearBase

    query_norm: RMSNorm | None
    key_norm: RMSNorm | None

    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    is_causal: bool = eqx.field(static=True)

    scale: float | None = eqx.field(static=True)
    sliding_window_size: int | None = eqx.field(static=True)

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
    def attention_type(self) -> AttentionType:
        return AttentionType.SLIDING_WINDOW if self.sliding_window_size is not None else AttentionType.GLOBAL

    def __post_init__(self) -> None:
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
        q_output_dim, k_output_dim, v_output_dim = self.qkv_projection.output_dims
        if q_output_dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"Query projection output dimension must be num_heads * head_dim"
                f" ({self.num_heads} * {self.head_dim} = {self.num_heads * self.head_dim}),"
                f" got {q_output_dim}",
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

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayer | None = None,
        return_updated_kv_cache: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> AttentionResult:
        queries, keys, values = vmap(self.qkv_projection, in_axes=0)(inputs)
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

        apply_positional_embeddings = vmap(positional_embeddings.apply, in_axes=1, out_axes=1)
        queries = apply_positional_embeddings(queries)
        keys = apply_positional_embeddings(keys)

        if kv_cache is None:
            updated_kv_cache = DynamicKVCacheLayer.init(keys, values, length=length_without_padding)
        else:
            updated_kv_cache = kv_cache.extend(keys, values, added_length=length_without_padding)

        num_suffix_tokens, _, _ = queries.shape
        mask = updated_kv_cache.attention_mask(num_suffix_tokens, self.is_causal, self.sliding_window_size)

        if self.config.logit_soft_cap is not None:
            attention_output = _soft_capped_attention_kernel(
                queries,
                updated_kv_cache.keys,
                updated_kv_cache.values,
                mask=mask,
                scale=self.scale,
                logit_soft_cap=self.config.logit_soft_cap,
            )
        else:
            attention_output = jax.nn.dot_product_attention(
                queries,
                updated_kv_cache.keys,
                updated_kv_cache.values,
                mask=mask,
                scale=self.scale,
            )
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.num_heads,
            head_channels=self.head_dim,
        )
        (result,) = vmap(self.out_projection, in_axes=0)(attention_output)

        if not return_updated_kv_cache:
            updated_kv_cache = None

        return AttentionResult(
            outputs=result,
            kv_cache=updated_kv_cache,
        )

    def init_static_kv_cache(self, capacity: int) -> StaticKVCacheLayer:
        return StaticKVCacheLayer.empty(capacity, self.num_groups, self.head_dim, self.activation_precision)

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:
        result = ParameterDict(
            qkv_projection=self.qkv_projection.export_weights(weight_layout),
            out_projection=self.out_projection.export_weights(weight_layout),
        )
        if self.query_norm is not None:
            result["query_norm"] = self.query_norm.export_weights(weight_layout)
        if self.key_norm is not None:
            result["key_norm"] = self.key_norm.export_weights(weight_layout)
        return result
