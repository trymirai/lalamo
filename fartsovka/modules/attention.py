from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from einops import einsum, rearrange, repeat
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Int

import os
import numpy as _np_debug

from fartsovka.common import ParameterDict

from .common import FartsovkaModule
from .kv_cache import KVCacheLayerSlice
from .linear import LinearBase, LinearConfig
from .rope import PositionalEmbeddings
from .utils import apply_soft_capping

__all__ = [
    "Attention",
    "AttentionConfig",
    "VisionSdpaAttention",
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


def _apply_sliding_window(
    mask: Bool[Array, "dst_tokens src_tokens"],
    local_window_size: int,
) -> Bool[Array, "dst_tokens src_tokens"]:
    dst_length, src_length = mask.shape
    dst_indices = jnp.arange(dst_length)[:, None]
    src_indices = jnp.arange(src_length)[None, :]
    return mask & (abs(dst_indices - src_indices) < local_window_size)


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


class AttentionOutput(NamedTuple):
    attention_output: Float[Array, "suffix_tokens channels"]
    kv_cache: KVCacheLayerSlice | None = None


@dataclass
class AttentionConfig:
    qkv_projection_config: LinearConfig
    out_projection_config: LinearConfig

    logit_soft_cap: float | None
    has_qkv_biases: bool
    has_out_biases: bool

    def random_init(
        self,
        model_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
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
        return Attention(
            self,
            qkv_projection=qkv_projection,
            out_projection=out_projection,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            scale=scale,
            sliding_window_size=sliding_window_size,
        )


class Attention(FartsovkaModule[AttentionConfig]):
    qkv_projection: LinearBase
    out_projection: LinearBase

    num_heads: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

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
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
        *,
        debug_prefix: str | None = None,
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
            all_values = jnp.concatenate([kv_cache.values, values], axis=0)
        else:
            all_keys = keys
            all_values = values

        if self.config.logit_soft_cap is not None:
            attention_output = _soft_capped_attention_kernel(
                queries,
                all_keys,
                all_values,
                mask=mask,
                scale=self.scale,
                logit_soft_cap=self.config.logit_soft_cap,
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
class VisionSdpaAttentionConfig:
    qkv_projection_config: LinearConfig
    out_projection_config: LinearConfig

    logit_soft_cap: float | None
    has_qkv_biases: bool
    has_out_biases: bool
 

    def random_init(
        self,
        model_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
    ) -> "VisionSdpaAttention":
        qkv = self.qkv_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(model_dim * 3,),  # Outputs concatenated Q, K, V
            has_biases=True,        # Matches nn.Linear(bias=True)
            key=key,
        )
        proj = self.out_projection_config.random_init(
            input_dim=model_dim,          # Input to proj is also dim (num_heads * head_dim)
            output_dims=(model_dim,),
            has_biases=True,        # Matches nn.Linear(bias=True)
            key=key,
        )

        return VisionSdpaAttention(
            self,
            qkv=qkv,
            proj=proj,
            num_heads=num_heads,
            head_dim=model_dim // num_heads,
        )


class VisionSdpaAttention(FartsovkaModule[VisionSdpaAttentionConfig]):
    qkv: LinearBase
    proj: LinearBase
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_length channels"],
        cu_seqlens: Int[Array, "num_segments_plus_1"] | None = None,
        attention_mask_external: Bool[Array, "... seq_length seq_length"] | None = None,
        position_embeddings: tuple[
            Float[Array, "seq_length head_dim"], Float[Array, "seq_length head_dim"]
        ] | None = None
    ) -> Float[Array, "seq_length channels"]:
        seq_length, dim = hidden_states.shape
        if dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"Input channels {dim} does not match num_heads*head_dim ({self.num_heads}*{self.head_dim})"
            )
        (qkv_out,) = vmap(self.qkv, in_axes=0)(hidden_states)  # Shape: "seq_length (3*dim)"

        qkv_reshaped = rearrange(
            qkv_out,
            "s (three h dh) -> three s h dh",
            three=3, h=self.num_heads, dh=self.head_dim
        )
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
        # q, k, v shapes: "seq_length num_heads head_dim"

        # Apply RoPE
        if position_embeddings is not None:
            cos_emb, sin_emb = position_embeddings
            if cos_emb.shape[-1] != self.head_dim or sin_emb.shape[-1] != self.head_dim:
                raise ValueError(
                    f"position_embeddings head_dim ({cos_emb.shape[-1]}) "
                    f"must match model head_dim ({self.head_dim})"
                )
            q, k = _apply_rotary_pos_emb_vision_jax(q, k, cos_emb, sin_emb)

        # Attention Mask
        final_attention_mask: Bool[Array, "... seq_length seq_length"] | None = None
        if attention_mask_external is not None:
            if attention_mask_external.ndim == 2: # "seq_length seq_length"
                final_attention_mask = attention_mask_external[None, :, :] # -> "1 seq_length seq_length"
            elif attention_mask_external.ndim == 3: # "1 seq_length seq_length" or "num_heads seq_length seq_length"
                final_attention_mask = attention_mask_external
            else:
                raise ValueError(
                    f"attention_mask_external has invalid ndim {attention_mask_external.ndim}, expected 2 or 3"
                )
        elif cu_seqlens is not None:
            final_attention_mask = _create_mask_from_cu_seqlens_jax(seq_length, cu_seqlens)
        # If both are None, final_attention_mask remains None (full attention)

        attn_output = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=final_attention_mask
        )
        # attn_output shape: "seq_length num_heads head_dim"

        attn_output_reshaped = rearrange(
            attn_output,
            "s h dh -> s (h dh)",
            h=self.num_heads, dh=self.head_dim
        )
        # attn_output_reshaped shape: "seq_length channels"

        # Final projection also using vmap
        (output,) = vmap(self.proj, in_axes=0)(attn_output_reshaped)  # Shape: "seq_length channels"

        return output

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            qkv_proj=self.qkv.export_weights(),
            out_proj=self.proj.export_weights(),
        )


# Helper functions for RoPE (ported from HF)
def _rotate_half_jax(x: Array) -> Array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

def _apply_rotary_pos_emb_vision_jax(
    q: Float[Array, "seq num_heads head_dim"],
    k: Float[Array, "seq num_heads head_dim"],
    cos_emb: Float[Array, "seq head_dim"],
    sin_emb: Float[Array, "seq head_dim"],
) -> tuple[Float[Array, "seq num_heads head_dim"], Float[Array, "seq num_heads head_dim"]]:
    """Applies Rotary Position Embedding to q and k."""
    # Unsqueeze cos/sin for broadcasting over heads dimension
    cos_emb_broadcast = rearrange(cos_emb, "s d -> s 1 d")
    sin_emb_broadcast = rearrange(sin_emb, "s d -> s 1 d")

    q_embed = (q * cos_emb_broadcast) + (_rotate_half_jax(q) * sin_emb_broadcast)
    k_embed = (k * cos_emb_broadcast) + (_rotate_half_jax(k) * sin_emb_broadcast)
    return q_embed, k_embed

# Helper for mask from cu_seqlens
def _create_mask_from_cu_seqlens_jax(
    seq_length: int,
    cu_seqlens: Int[Array, "num_segments_plus_1"], # e.g., [0, len1, len1+len2, ...]
    dtype: jnp.dtype = jnp.bool_
) -> Bool[Array, "1 seq_length seq_length"]:
    mask = jnp.zeros((1, seq_length, seq_length), dtype=dtype)
    num_segments = cu_seqlens.shape[0] - 1

    if num_segments < 0:
        raise ValueError("cu_seqlens must have at least one element.")

    if seq_length == 0:
        return mask
    
    row_indices = jnp.arange(seq_length)
    col_indices = jnp.arange(seq_length)
    
    row_indices = row_indices.reshape(1, seq_length, 1).repeat(seq_length, axis=2)
    col_indices = col_indices.reshape(1, 1, seq_length).repeat(seq_length, axis=1)
    
    def create_segment_mask(segment_idx, seqlens_array):
        start_idx = seqlens_array[segment_idx]
        end_idx = seqlens_array[segment_idx + 1]
        
        segment_mask = (row_indices >= start_idx) & (row_indices < end_idx) & \
                       (col_indices >= start_idx) & (col_indices < end_idx)
        
        return segment_mask
    
    result_mask = jnp.zeros((1, seq_length, seq_length), dtype=dtype)
    
    for i in range(num_segments):
        segment_mask = create_segment_mask(i, cu_seqlens)
        result_mask = result_mask | segment_mask
    
    return result_mask
