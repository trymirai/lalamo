from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
from einops import einsum, rearrange
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, DTypeLike, Float, Int, Key

from lalamo.initializer import Initializer
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.utils import apply_soft_capping, vmap_with_dequant_key

from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, StaticKVCacheLayer

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
class AttentionConfig(TokenMixerConfig):
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
    # Scale-free RMS normalization on values
    normalize_values: bool = False

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
            sinks = initializer.zeros((self.num_heads,))
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
    def model_dim(self) -> int:
        return self.qkv_projection.input_dim

    @property
    def group_size(self) -> int:
        return self.config.num_heads // self.config.num_groups

    @property
    def use_sliding_window(self) -> bool:
        return self.config.sliding_window_size is not None

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        if not self.config.use_rope:
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
        forward_pass_config: MixerForwardPassConfig | None = None,
        *,
        dequant_key: Key[Array, ""],
    ) -> AttentionResult:
        if forward_pass_config is None:
            forward_pass_config = MixerForwardPassConfig()
        qkv_dequant_key, gate_dequant_key, out_dequant_key = jax.random.split(dequant_key, 3)
        queries, keys, values = vmap_with_dequant_key(
            partial(self.qkv_projection, forward_pass_config=forward_pass_config.arrays),
            inputs,
            dequant_key=qkv_dequant_key,
        )
        if self.gate_projection is not None:
            (gate,) = vmap_with_dequant_key(
                partial(self.gate_projection, forward_pass_config=forward_pass_config.arrays),
                inputs,
                dequant_key=gate_dequant_key,
            )
        else:
            gate = None

        queries = rearrange(
            queries,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        keys = rearrange(
            keys,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_groups,
            head_channels=self.config.head_dim,
        )
        values = rearrange(
            values,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_groups,
            head_channels=self.config.head_dim,
        )

        if self.query_norm is not None:
            queries = vmap(vmap(self.query_norm))(queries)
        if self.key_norm is not None:
            keys = vmap(vmap(self.key_norm))(keys)
        if self.config.normalize_values:
            values = _rms_normalize(values, eps=1e-6)

        if positional_embeddings is not None:
            apply_positional_embeddings = vmap(positional_embeddings.apply, in_axes=1, out_axes=1)
            queries = apply_positional_embeddings(queries)
            keys = apply_positional_embeddings(keys)

        prefix_length = 0 if state is None else state.current_prefix_length()
        if state is None:
            updated_state = DynamicKVCacheLayer.init(self.has_sinks, keys, values, length=length_without_padding)
        else:
            updated_state = state.extend(keys, values, added_length=length_without_padding)

        num_suffix_tokens, _, _ = queries.shape
        mask = updated_state.attention_mask(
            num_suffix_tokens,
            self.config.is_causal,
            length_without_padding,
            self.config.sliding_window_size,
        )
        if self.sinks is not None:
            sink_bias = jnp.zeros((self.config.num_heads, *mask.shape), dtype=queries.dtype)
            sink_bias = sink_bias.at[:, :, 0].set(self.sinks[:, None])
        else:
            sink_bias = None

        if self.config.logit_soft_cap is not None:
            attention_output = _soft_capped_attention_kernel(
                queries,
                updated_state.keys,
                updated_state.values,
                mask=mask,
                scale=self.config.scale,
                logit_soft_cap=self.config.logit_soft_cap,
            )
        else:
            attention_output = jax.nn.dot_product_attention(
                queries,
                updated_state.keys,
                updated_state.values,
                bias=sink_bias,
                mask=mask,
                scale=self.config.scale,
            )
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        if gate is not None:
            attention_output = attention_output * jax.nn.sigmoid(gate)
        (result,) = vmap_with_dequant_key(
            partial(self.out_projection, forward_pass_config=forward_pass_config.arrays),
            attention_output,
            dequant_key=out_dequant_key,
        )

        if not return_updated_state:
            updated_state = None

        return AttentionResult(
            outputs=result,
            state=updated_state,
        )

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StaticKVCacheLayer:
        return StaticKVCacheLayer.init(
            self.has_sinks,
            capacity,
            self.config.num_groups,
            self.config.head_dim,
            dtype,
        )
