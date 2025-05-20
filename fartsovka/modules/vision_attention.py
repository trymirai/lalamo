from dataclasses import dataclass

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Int

from fartsovka.common import ParameterDict

from .common import FartsovkaModule
from .linear import LinearBase, LinearConfig
__all__ = [
    "VisionAttention"
]

@dataclass
class VisionAttentionConfig:
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
    ) -> "VisionAttention":
        qkv = self.qkv_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(model_dim * 3,),  
            has_biases=True,       
            key=key,
        )
        proj = self.out_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(model_dim,),
            has_biases=True,
            key=key,
        )

        return VisionAttention(
            self,
            qkv=qkv,
            proj=proj,
            num_heads=num_heads,
            head_dim=model_dim // num_heads,
        )


class VisionAttention(FartsovkaModule[VisionAttentionConfig]):
    qkv: LinearBase
    proj: LinearBase
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_length channels"],
        cu_seqlens: Int[Array, "num_segments_plus_1"] | None = None,
        position_embeddings: tuple[
            Float[Array, "seq_length head_dim"], Float[Array, "seq_length head_dim"]
        ] | None = None
    ) -> Float[Array, "seq_length channels"]:
        seq_length, dim = hidden_states.shape
        if dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"Input channels {dim} does not match num_heads*head_dim ({self.num_heads}*{self.head_dim})"
            )
        (qkv_out,) = vmap(self.qkv, in_axes=0)(hidden_states)  

        qkv_reshaped = rearrange(
            qkv_out,
            "s (three h dh) -> three s h dh",
            three=3, h=self.num_heads, dh=self.head_dim
        )
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]

        if position_embeddings is not None:
            cos_emb, sin_emb = position_embeddings
            if cos_emb.shape[-1] != self.head_dim or sin_emb.shape[-1] != self.head_dim:
                raise ValueError(
                    f"position_embeddings head_dim ({cos_emb.shape[-1]}) "
                    f"must match model head_dim ({self.head_dim})"
                )
            q, k = apply_rotary_pos_emb_vision(q, k, cos_emb, sin_emb)

        final_attention_mask: Bool[Array, "... seq_length seq_length"] | None = None
        if cu_seqlens is not None:
            final_attention_mask = _create_mask_from_cu_seqlens(seq_length, cu_seqlens)

        attn_output = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=final_attention_mask
        )

        attn_output_reshaped = rearrange(
            attn_output,
            "s h dh -> s (h dh)",
            h=self.num_heads, dh=self.head_dim
        )

        (output,) = vmap(self.proj, in_axes=0)(attn_output_reshaped)

        return output

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            qkv_proj=self.qkv.export_weights(),
            out_proj=self.proj.export_weights(),
        )


def rotate_half(x: Array) -> Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb_vision(
    q: Float[Array, "seq num_heads head_dim"],
    k: Float[Array, "seq num_heads head_dim"],
    cos_emb: Float[Array, "seq head_dim"],
    sin_emb: Float[Array, "seq head_dim"],
) -> tuple[Float[Array, "seq num_heads head_dim"], Float[Array, "seq num_heads head_dim"]]:
    cos_emb_broadcast = rearrange(cos_emb, "s d -> s 1 d")
    sin_emb_broadcast = rearrange(sin_emb, "s d -> s 1 d")

    q_embed = (q * cos_emb_broadcast) + (rotate_half(q) * sin_emb_broadcast)
    k_embed = (k * cos_emb_broadcast) + (rotate_half(k) * sin_emb_broadcast)
    return q_embed, k_embed

def _create_mask_from_cu_seqlens(
    seq_length: int,
    cu_seqlens: Int[Array, "num_segments_plus_1"],
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
