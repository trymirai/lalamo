import warnings
from dataclasses import dataclass

import equinox as eqx
import jax
import tokamax
from einops import einsum, rearrange
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import (
    Normalization,
    NormalizationConfig,
    NormalizationForwardPassConfig,
    NormalizationImplementation,
)
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixer import (
    AttentionImplementation,
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.utils import apply_soft_capping, call_vmapped, call_vmapped_twice

from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, StaticKVCacheLayer

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
]


def _repeat_kv(
    keys_or_values: Float[Array, "tokens groups channels"],
    group_size: int,
) -> Float[Array, "tokens heads channels"]:
    if group_size == 1:
        return keys_or_values
    return jnp.repeat(keys_or_values, group_size, axis=1)


def _rms_normalize(
    inputs: Float[Array, "... channels"],
    eps: float,
    forward_pass_config: NormalizationForwardPassConfig,
) -> Float[Array, "... channels"]:
    match forward_pass_config.implementation:
        case NormalizationImplementation.JAX:
            upcasted_inputs = inputs.astype(jnp.float32)
            variance = jnp.mean(jnp.square(upcasted_inputs), axis=-1, keepdims=True)
            return (upcasted_inputs * jax.lax.rsqrt(variance + eps)).astype(inputs.dtype)

        case NormalizationImplementation.TOKAMAX:
            return tokamax.layer_norm(
                inputs,
                scale=None,
                offset=None,
                epsilon=eps,
                subtract_mean=False,
            ).astype(inputs.dtype)


def _soft_capped_attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    *,
    bias: Float[Array, "heads dst_tokens src_tokens"] | None,
    mask: Bool[Array, "dst_tokens src_tokens"] | None,
    scale: float | None,
    logit_soft_cap: float | None,
) -> Float[Array, "dst_tokens heads head_channels"]:
    original_dtype = queries.dtype
    attention_dtype = jnp.float32
    queries = queries.astype(attention_dtype)
    keys = keys.astype(attention_dtype)
    values = values.astype(attention_dtype)
    if bias is not None:
        bias = bias.astype(attention_dtype)

    if logit_soft_cap is None:
        return jax.nn.dot_product_attention(
            queries,
            keys,
            values,
            bias=bias,
            mask=mask,
            scale=scale,
        ).astype(original_dtype)

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
    if scale is None:
        scale_val = head_dim**-0.5
    else:
        scale_val = float(scale)
    attention_logits = attention_logits * scale_val
    attention_logits = apply_soft_capping(attention_logits, logit_soft_cap)
    if bias is not None:
        attention_logits = attention_logits + bias
    if mask is not None:
        attention_logits = jnp.where(
            mask,
            attention_logits,
            jnp.array(float("-inf"), dtype=attention_logits.dtype),
        )
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    return einsum(
        attention_weights,
        values,
        "heads dst_tokens src_tokens, src_tokens heads channels -> dst_tokens heads channels",
    ).astype(original_dtype)


def _stable_reduction_attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    *,
    bias: Float[Array, "heads dst_tokens src_tokens"] | None,
    mask: Bool[Array, "dst_tokens src_tokens"] | None,
    scale: float | None,
    logit_soft_cap: float | None,
    tile_size: int,
    accumulation_dtype: DTypeLike | None,
) -> Float[Array, "dst_tokens heads head_channels"]:
    if tile_size < 1:
        raise ValueError("attention_tile_size must be at least 1.")

    original_dtype = queries.dtype
    accumulation_dtype = original_dtype if accumulation_dtype is None else accumulation_dtype

    num_queries, num_heads, head_dim = queries.shape
    num_keys, num_groups, _ = keys.shape
    group_size = num_heads // num_groups

    if scale is None:
        scale = head_dim**-0.5
    else:
        scale = float(scale)

    if mask is None:
        mask = jnp.ones((num_queries, num_keys), dtype=jnp.bool_)

    if group_size > 1:
        keys = _repeat_kv(keys, group_size)
        values = _repeat_kv(values, group_size)

    pad_len = (-num_keys) % tile_size
    num_tiles = (num_keys + pad_len) // tile_size

    keys = jnp.pad(keys, [(0, pad_len), (0, 0), (0, 0)])
    values = jnp.pad(values, [(0, pad_len), (0, 0), (0, 0)])

    queries = rearrange(queries, "tokens heads channels -> heads tokens channels").astype(accumulation_dtype)
    key_tiles = rearrange(
        keys,
        "(tiles tokens) heads channels -> tiles heads tokens channels",
        tiles=num_tiles,
        tokens=tile_size,
    )
    value_tiles = rearrange(
        values,
        "(tiles tokens) heads channels -> tiles heads tokens channels",
        tiles=num_tiles,
        tokens=tile_size,
    )
    mask_tiles = rearrange(
        jnp.pad(mask, [(0, 0), (0, pad_len)], constant_values=False),
        "queries (tiles tokens) -> tiles queries tokens",
        tiles=num_tiles,
        tokens=tile_size,
    )
    if bias is None:
        bias_tiles = None
    else:
        bias_tiles = rearrange(
            jnp.pad(bias, [(0, 0), (0, 0), (0, pad_len)]),
            "heads queries (tiles tokens) -> tiles heads queries tokens",
            tiles=num_tiles,
            tokens=tile_size,
        ).astype(accumulation_dtype)

    scores = einsum(
        queries,
        key_tiles.astype(accumulation_dtype),
        "heads queries channels, tiles heads tokens channels -> tiles heads queries tokens",
    )
    scores = scale * scores
    if logit_soft_cap is not None:
        scores = apply_soft_capping(scores, logit_soft_cap)
    if bias_tiles is not None:
        scores = scores + bias_tiles
    scores = jnp.where(
        mask_tiles[:, None, :, :],
        scores,
        jnp.array(float("-inf"), dtype=accumulation_dtype),
    )

    tile_max = jnp.max(scores, axis=-1)
    safe_tile_max = jnp.where(jnp.isneginf(tile_max), 0.0, tile_max)
    exp_scores = jnp.exp(scores - safe_tile_max[..., None])
    tile_sum = jnp.sum(exp_scores, axis=-1)
    tile_output = einsum(
        exp_scores,
        value_tiles.astype(accumulation_dtype),
        "tiles heads queries tokens, tiles heads tokens channels -> tiles heads queries channels",
    )

    def combine(left: tuple, right: tuple) -> tuple:
        left_max, left_sum, left_output = left
        right_max, right_sum, right_output = right
        new_max = jnp.maximum(left_max, right_max)
        safe_new_max = jnp.where(jnp.isneginf(new_max), 0.0, new_max)
        left_correction = jnp.exp(left_max - safe_new_max)
        right_correction = jnp.exp(right_max - safe_new_max)
        return (
            new_max,
            left_correction * left_sum + right_correction * right_sum,
            left_correction[..., None] * left_output + right_correction[..., None] * right_output,
        )

    _, final_sum, final_output = jax.lax.associative_scan(
        combine,
        (tile_max, tile_sum, tile_output),
    )
    result = final_output[-1] / final_sum[-1, ..., None]
    return rearrange(result, "heads queries channels -> queries heads channels").astype(original_dtype)


def _attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    *,
    bias: Float[Array, "heads dst_tokens src_tokens"] | None,
    mask: Bool[Array, "dst_tokens src_tokens"] | None,
    scale: float | None,
    logit_soft_cap: float | None,
    forward_pass_config: MixerForwardPassConfig,
) -> Float[Array, "dst_tokens heads head_channels"]:
    def tokamax_attention() -> Float[Array, "dst_tokens heads head_channels"]:
        if bias is not None and logit_soft_cap is not None:
            raise RuntimeError("Tokamax attention does not support logit soft-capping with additive bias.")
        return tokamax.dot_product_attention(
            queries,
            keys,
            values,
            bias=bias,
            mask=mask,
            scale=scale,
            logits_soft_cap=logit_soft_cap,
        ).astype(queries.dtype)

    match forward_pass_config.attention_implementation:
        case AttentionImplementation.STANDARD:
            return _soft_capped_attention_kernel(
                queries,
                keys,
                values,
                bias=bias,
                mask=mask,
                scale=scale,
                logit_soft_cap=logit_soft_cap,
            )
        case AttentionImplementation.CUDNN:
            head_dim = queries.shape[-1]
            if head_dim > 128 or head_dim % 8 != 0:
                warnings.warn(
                    "cuDNN attention requires head_dim <= 128 and divisible by 8; "
                    f"got head_dim={head_dim}. Falling back to standard (jax/xla) attention.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return _soft_capped_attention_kernel(
                    queries,
                    keys,
                    values,
                    bias=bias,
                    mask=mask,
                    scale=scale,
                    logit_soft_cap=logit_soft_cap,
                )
            if logit_soft_cap is not None:
                raise RuntimeError("cuDNN attention does not support logit soft-capping.")
            if mask is not None:
                mask = jnp.broadcast_to(mask, (queries.shape[1], *mask.shape))
            original_dtype = queries.dtype
            attention_dtype = jnp.float32
            return jax.nn.dot_product_attention(
                queries.astype(attention_dtype),
                keys.astype(attention_dtype),
                values.astype(attention_dtype),
                bias=None if bias is None else bias.astype(attention_dtype),
                mask=mask,
                scale=scale,
                implementation="cudnn",
            ).astype(original_dtype)
        case AttentionImplementation.TOKAMAX:
            return tokamax_attention()
        case AttentionImplementation.STABLE_REDUCTION:
            return _stable_reduction_attention_kernel(
                queries,
                keys,
                values,
                bias=bias,
                mask=mask,
                scale=scale,
                logit_soft_cap=logit_soft_cap,
                tile_size=forward_pass_config.attention_tile_size,
                accumulation_dtype=forward_pass_config.attention_accumulation_dtype,
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
    is_kv_sharing: bool = False

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "Attention":
        q_output_dim = self.num_heads * self.head_dim
        if self.is_kv_sharing:
            output_dims = (q_output_dim,)
        else:
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

        if self.key_norm_config is not None and not self.is_kv_sharing:
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
            sharding_config=initializer.sharding_config,
            qkv_projection=qkv_projection,
            gate_projection=gate_projection,
            out_projection=out_projection,
            query_norm=query_norm,
            key_norm=key_norm,
            sinks=sinks,
        )


class Attention(TokenMixerBase[AttentionConfig, KVCacheLayer]):
    qkv_projection: Linear
    gate_projection: Linear | None
    out_projection: Linear

    query_norm: Normalization | None
    key_norm: Normalization | None

    sinks: Float[Array, " heads"] | None

    @property
    def model_dim(self) -> int:
        return self.qkv_projection.input_dim

    @property
    def use_sliding_window(self) -> bool:
        return self.config.sliding_window_size is not None

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        if self.use_sliding_window:
            return PositionalEmbeddingSelector.LOCAL
        return PositionalEmbeddingSelector.GLOBAL

    @property
    def has_sinks(self) -> bool:
        return self.sinks is not None

    @eqx.filter_jit
    def _prepare_heads(
        self,
        projection: Float[Array, "tokens channels"],
        num_heads: int,
        norm: Normalization | None,
        positional_embeddings: PositionalEmbeddings | None,
        forward_pass_config: MixerForwardPassConfig,
    ) -> Float[Array, "tokens heads head_channels"]:
        heads = rearrange(
            projection,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=num_heads,
            head_channels=self.config.head_dim,
        )
        if norm is not None:
            heads = call_vmapped_twice(
                norm,
                heads,
                forward_pass_config=forward_pass_config.normalization_forward_pass_config,
            )
        if positional_embeddings is not None:
            heads = call_vmapped(positional_embeddings.apply, heads, in_axes=1, out_axes=1)
        return heads

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: KVCacheLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        reuse_cache: bool = False,
        *,
        keychain: Keychain,
    ) -> AttentionResult:
        qkv_keychain, gate_keychain, out_keychain = keychain.split(3)
        assert reuse_cache == self.config.is_kv_sharing, "reuse_cache must match AttentionConfig.is_kv_sharing"
        projections = call_vmapped(
            self.qkv_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=qkv_keychain,
        )
        queries = projections[0]
        if self.gate_projection is not None:
            (gate,) = call_vmapped(
                self.gate_projection,
                inputs,
                forward_pass_config=forward_pass_config.matmul_config,
                keychain=gate_keychain,
            )
        else:
            gate = None

        queries = self._prepare_heads(
            queries, self.config.num_heads, self.query_norm, positional_embeddings, forward_pass_config
        )

        num_suffix_tokens, _, _ = queries.shape
        if reuse_cache:
            if state is None:
                raise ValueError("a KV-sharing layer must receive the source layer's cache as `state`")
            prefix_length = state.current_prefix_length() - num_suffix_tokens
            updated_state = state
        else:
            _, keys, values = projections
            prefix_length = 0 if state is None else state.current_prefix_length()
            keys = self._prepare_heads(
                keys, self.config.num_groups, self.key_norm, positional_embeddings, forward_pass_config
            )
            values = rearrange(
                values,
                "tokens (groups head_channels) -> tokens groups head_channels",
                groups=self.config.num_groups,
                head_channels=self.config.head_dim,
            )
            if self.config.normalize_values:
                values = _rms_normalize(
                    values,
                    eps=1e-6,
                    forward_pass_config=forward_pass_config.normalization_forward_pass_config,
                )
            if state is None:
                updated_state = DynamicKVCacheLayer.init(
                    self.has_sinks, keys.astype(values.dtype), values, length=length_without_padding
                )
            else:
                updated_state = state.extend(keys, values, added_length=length_without_padding)

        queries = queries.astype(updated_state.keys.dtype)
        if attention_parent_indices is not None:
            mask = updated_state.tree_attention_mask(prefix_length, attention_parent_indices)
        else:
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

        attention_output = _attention_kernel(
            queries,
            updated_state.keys,
            updated_state.values,
            bias=sink_bias,
            mask=mask,
            scale=self.config.scale,
            logit_soft_cap=self.config.logit_soft_cap,
            forward_pass_config=forward_pass_config,
        )
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        if gate is not None:
            attention_output = attention_output * jax.nn.sigmoid(gate)
        (result,) = call_vmapped(
            self.out_projection,
            attention_output,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain,
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
            self.sharding_config,
        )
