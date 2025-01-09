from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from einops import rearrange
from jax import nn, vmap
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .kv_cache import KVCacheLayerSlice
from .linear import LinearBase, LinearFactoryBase
from .rope import PositionalEmbeddings

__all__ = ["Attention", "AttentionFactory", "AttentionBase", "AttentionFactoryBase"]


class AttentionOutput(NamedTuple):
    attention_output: Float[Array, "suffix_tokens channels"]
    kv_cache: KVCacheLayerSlice | None = None


class AttentionBase(eqx.Module):
    model_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

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


class Attention[QKVProjType: LinearBase, OutProjType: LinearBase](AttentionBase):
    qkv_projection: QKVProjType
    out_projection: OutProjType

    def __init__(
        self,
        qkv_projection_factory: LinearFactoryBase[QKVProjType],
        out_projection_factory: LinearFactoryBase[OutProjType],
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        qkv_key, out_key = jax.random.split(key)
        self.qkv_projection = qkv_projection_factory(
            model_dim,
            (num_heads + 2 * num_groups) * head_dim,
            key=qkv_key,
        )
        self.out_projection = out_projection_factory(
            num_heads * head_dim,
            model_dim,
            key=out_key,
        )
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim

    def __call__(
        self,
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> AttentionOutput:
        qkv = vmap(self.qkv_projection, in_axes=0)(x)
        slice_indices = [
            self.num_heads * self.head_dim,
            self.num_heads * self.head_dim + self.num_groups * self.head_dim,
        ]

        queries, keys, values = jnp.split(qkv, slice_indices, axis=-1)
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

        attention_output = nn.dot_product_attention(queries, all_keys, all_values, mask=mask)
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.num_heads,
            head_channels=self.head_dim,
        )
        result = vmap(self.out_projection, in_axes=0)(attention_output)

        if return_updated_kv_cache:
            updated_kv_cache = KVCacheLayerSlice(keys=all_keys, values=all_values)
        else:
            updated_kv_cache = None
        return AttentionOutput(
            attention_output=result,
            kv_cache=updated_kv_cache,
        )


@dataclass
class AttentionFactoryBase[AttentionType: AttentionBase]:
    def __call__(
        self,
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> AttentionType:
        raise NotImplementedError


@dataclass
class AttentionFactory[QKVProjType: LinearBase, OutProjType: LinearBase](
    AttentionFactoryBase[Attention[QKVProjType, OutProjType]],
):
    qkv_projection_factory: LinearFactoryBase[QKVProjType]
    out_projection_factory: LinearFactoryBase[OutProjType]

    def __call__(
        self,
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> Attention[QKVProjType, OutProjType]:
        return Attention(
            self.qkv_projection_factory,
            self.out_projection_factory,
            model_dim,
            num_heads,
            num_groups,
            head_dim,
            key=key,
        )
