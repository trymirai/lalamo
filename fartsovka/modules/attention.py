from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from einops import einsum, rearrange, repeat
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .common import DummyUnionMember, FartsovkaModule, ParameterDict, register_config_union
from .kv_cache import KVCacheLayerSlice
from .linear import AbstractLinear, AbstractLinearConfig, LinearConfigType
from .rope import PositionalEmbeddings
from .utils import apply_soft_capping

__all__ = [
    "AbstractAttention",
    "AbstractAttentionConfig",
    "Attention",
    "AttentionConfig",
    "AttentionConfigType",
]


class AttentionOutput(NamedTuple):
    attention_output: Float[Array, "suffix_tokens channels"]
    kv_cache: KVCacheLayerSlice | None = None


class AbstractAttention(FartsovkaModule):
    model_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    scale: float | None = eqx.field(static=True)
    logit_soft_cap: float | None = eqx.field(static=True)
    sliding_window_size: int | None = eqx.field(static=True)

    @property
    def group_dim(self) -> int:
        return self.num_heads // self.num_groups

    def __call__(
        self,
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> AttentionOutput:
        raise NotImplementedError


@dataclass
class AbstractAttentionConfig[AttentionType: AbstractAttention]:
    def __call__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        scale: float | None,
        logit_soft_cap: float | None,
        sliding_window_size: int | None,
        use_qkv_bias: bool,
        use_out_bias: bool,
        key: PRNGKeyArray,
    ) -> AttentionType:
        raise NotImplementedError


def _repeat_kv(
    keys_or_values: Float[Array, "tokens groups channels"],
    group_size: int,
) -> Float[Array, "tokens groups*group_size channels"]:
    return repeat(
        keys_or_values,
        "tokens groups channels -> tokens (groups group_size) channels",
        group_size=group_size,
    )


def _apply_sliding_window(
    mask: Bool[Array, "dst_tokens src_tokens"],
    local_window_size: int,
) -> Bool[Array, "dst_tokens src_tokens"]:
    dst_length, src_length = mask.shape
    dst_indices = jnp.arange(dst_length)[:, None]
    src_indices = jnp.arange(src_length)[None, :]
    return mask & (dst_indices - src_indices < local_window_size)


def _soft_capped_attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    mask: Bool[Array, "dst_tokens src_tokens"] | None,
    local_window_size: int | None,
    scale: float | None,
    logit_soft_cap: float,
) -> Float[Array, "dst_tokens heads head_channels"]:
    dst_length, num_heads, head_dim = queries.shape
    src_length, num_groups, _ = keys.shape
    if scale is None:
        scale = head_dim**-0.5
    if local_window_size is not None:
        if mask is None:
            mask = jnp.ones((dst_length, src_length), dtype=jnp.bool_)
        mask = _apply_sliding_window(mask, local_window_size)
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


class Attention[QKVProjType: AbstractLinear, OutProjType: AbstractLinear](AbstractAttention):
    qkv_projection: QKVProjType
    out_projection: OutProjType

    def __init__(
        self,
        *,
        qkv_projection_config: AbstractLinearConfig[QKVProjType],
        out_projection_config: AbstractLinearConfig[OutProjType],
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        scale: float | None,
        logit_soft_cap: float | None,
        sliding_window_size: int | None,
        use_qkv_bias: bool,
        use_out_bias: bool,
        key: PRNGKeyArray,
    ) -> None:
        qkv_key, out_key = jax.random.split(key)
        super().__init__(model_dim, num_heads, num_groups, head_dim, scale, logit_soft_cap, sliding_window_size)
        self.qkv_projection = qkv_projection_config(
            model_dim,
            (num_heads * head_dim, num_groups * head_dim, num_groups * head_dim),
            key=qkv_key,
            use_bias=use_qkv_bias,
        )
        self.out_projection = out_projection_config(
            num_heads * head_dim,
            (model_dim,),
            key=out_key,
            use_bias=use_out_bias,
        )

    def __call__(
        self,
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> AttentionOutput:
        queries, keys, values = vmap(self.qkv_projection, in_axes=0)(x)
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

        apply_positional_embeddings = vmap(positional_embeddings.apply, in_axes=1, out_axes=1)
        queries = apply_positional_embeddings(queries)
        keys = apply_positional_embeddings(keys)

        if kv_cache is not None:
            all_keys = jnp.concatenate([kv_cache.keys, keys], axis=0)
            all_values = jnp.concatenate([kv_cache.values, values], axis=0)  # noqa: PD011
        else:
            all_keys = keys
            all_values = values

        if self.logit_soft_cap is not None:
            attention_output = _soft_capped_attention_kernel(
                queries,
                all_keys,
                all_values,
                mask=mask,
                scale=self.scale,
                logit_soft_cap=self.logit_soft_cap,
                local_window_size=self.sliding_window_size,
            )
        else:
            attention_output = jax.nn.dot_product_attention(
                queries,
                all_keys,
                all_values,
                mask=mask,
                scale=self.scale,
                local_window_size=self.sliding_window_size,
            )
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.num_heads,
            head_channels=self.head_dim,
        )
        (result,) = vmap(self.out_projection, in_axes=0)(attention_output)

        if return_updated_kv_cache:
            updated_kv_cache = KVCacheLayerSlice(keys=all_keys, values=all_values)
        else:
            updated_kv_cache = None
        return AttentionOutput(
            attention_output=result,
            kv_cache=updated_kv_cache,
        )

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            qkv_proj=self.qkv_projection.export_weights(),
            out_proj=self.out_projection.export_weights(),
        )


@dataclass
class AttentionConfig[QKVProjType: AbstractLinear, OutProjType: AbstractLinear](
    AbstractAttentionConfig[Attention[QKVProjType, OutProjType]],
):
    qkv_projection_config: LinearConfigType
    out_projection_config: LinearConfigType

    def __call__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        scale: float | None,
        logit_soft_cap: float | None,
        sliding_window_size: int | None,
        use_qkv_bias: bool,
        use_out_bias: bool,
        key: PRNGKeyArray,
    ) -> Attention[QKVProjType, OutProjType]:
        return Attention(
            qkv_projection_config=self.qkv_projection_config,  # type: ignore
            out_projection_config=self.out_projection_config,  # type: ignore
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            scale=scale,
            logit_soft_cap=logit_soft_cap,
            sliding_window_size=sliding_window_size,
            use_qkv_bias=use_qkv_bias,
            use_out_bias=use_out_bias,
            key=key,
        )


AttentionConfigType = AttentionConfig | DummyUnionMember


register_config_union(AttentionConfigType)
