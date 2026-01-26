from dataclasses import dataclass

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, Int

from lalamo.modules.common import Initializer, PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfigBase
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .state import DynamicKVCacheLayer, KVCacheLayer, StaticKVCacheLayer

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
]


AttentionResult = TokenMixerResult[KVCacheLayer]


@dataclass(frozen=True)
class AttentionConfig(TokenMixerConfigBase):
    qkv_projection_config: LinearConfigBase
    out_projection_config: LinearConfigBase

    query_norm_config: NormalizationConfig | None
    key_norm_config: NormalizationConfig | None

    num_heads: int
    num_groups: int
    head_dim: int
    is_causal: bool
    scale: float | None
    sliding_window_size: int | None

    has_sinks: bool
    has_qkv_biases: bool
    has_out_biases: bool

    @property
    def rope_dim(self) -> int:
        return self.head_dim

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "Attention":
        qkv_projection = self.qkv_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(
                self.num_heads * self.head_dim,
                self.num_groups * self.head_dim,
                self.num_groups * self.head_dim,
            ),
            has_biases=self.has_qkv_biases,
        )
        out_projection = self.out_projection_config.init(
            initializer,
            self.num_heads * self.head_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
        )

        if self.query_norm_config is not None:
            query_norm = self.query_norm_config.init(initializer, self.head_dim)
        else:
            query_norm = None

        if self.key_norm_config is not None:
            key_norm = self.key_norm_config.init(initializer, self.head_dim)
        else:
            key_norm = None

        if self.has_sinks:
            sinks = initializer.zeros((self.num_heads,), qkv_projection.activation_precision)
        else:
            sinks = None

        return Attention(
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
        )


class Attention(TokenMixerBase[KVCacheLayer]):
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
