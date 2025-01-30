from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from einops import rearrange
from jax import nn, vmap
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .common import DummyUnionMember, FartsovkaModule, ParameterDict, register_config_union
from .kv_cache import KVCacheLayerSlice
from .linear import AbstractLinear, AbstractLinearConfig, LinearConfigType
from .rope import PositionalEmbeddings

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


class Attention[QKVProjType: AbstractLinear, OutProjType: AbstractLinear](AbstractAttention):
    use_qkv_bias: bool = eqx.field(static=True)
    use_out_bias: bool = eqx.field(static=True)

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
        sliding_window_size: int | None,
        use_qkv_bias: bool,
        use_out_bias: bool,
        key: PRNGKeyArray,
    ) -> None:
        qkv_key, out_key = jax.random.split(key)
        super().__init__(model_dim, num_heads, num_groups, head_dim, sliding_window_size)
        self.use_qkv_bias = use_qkv_bias
        self.use_out_bias = use_out_bias
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

        attention_output = nn.dot_product_attention(
            queries,
            all_keys,
            all_values,
            mask=mask,
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
class AbstractAttentionConfig[AttentionType: AbstractAttention]:
    def __call__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        sliding_window_size: int | None,
        use_qkv_bias: bool,
        use_out_bias: bool,
        key: PRNGKeyArray,
    ) -> AttentionType:
        raise NotImplementedError


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
            sliding_window_size=sliding_window_size,
            use_qkv_bias=use_qkv_bias,
            use_out_bias=use_out_bias,
            key=key,
        )


AttentionConfigType = AttentionConfig | DummyUnionMember


register_config_union(AttentionConfigType)
