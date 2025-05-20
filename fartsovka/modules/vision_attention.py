from dataclasses import dataclass

import equinox as eqx
import jax
from einops import rearrange, repeat
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from fartsovka.common import ParameterDict

from .common import FartsovkaModule
from .linear import LinearBase, LinearConfig
from .rope import PositionalEmbeddings

__all__ = [
    "VisionAttention",
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
        qkv_proj = self.qkv_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(model_dim * 3,),
            has_biases=True,
            key=key,
        )
        output_proj = self.out_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(model_dim,),
            has_biases=True,
            key=key,
        )

        return VisionAttention(
            self,
            qkv_proj=qkv_proj,
            output_proj=output_proj,
            num_heads=num_heads,
            head_dim=model_dim // num_heads,
        )


class VisionAttention(FartsovkaModule[VisionAttentionConfig]):
    qkv_proj: LinearBase
    output_proj: LinearBase
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_length channels"],
        cumulative_seqlens: Int[Array, "num_segments_plus_1"] | None = None,
        position_embeddings: PositionalEmbeddings | None = None,
    ) -> Float[Array, "seq_length channels"]:
        seq_length, dim = hidden_states.shape
        if dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"Input channels {dim} does not match num_heads*head_dim ({self.num_heads}*{self.head_dim})",
            )
        (qkv_out,) = vmap(self.qkv_proj, in_axes=0)(hidden_states)

        qkv_reshaped = rearrange(
            qkv_out,
            "s (three num_heads head_dim) -> three s num_heads head_dim",
            three=3, num_heads=self.num_heads, head_dim=self.head_dim,
        )
        queries, keys, values = qkv_reshaped

        if position_embeddings is not None:
            cos_emb = position_embeddings.cosines
            sin_emb = position_embeddings.sines
            if cos_emb.shape[-1] != self.head_dim or sin_emb.shape[-1] != self.head_dim:
                raise ValueError(
                    f"position_embeddings head_dim ({cos_emb.shape[-1]}) "
                    f"must match model head_dim ({self.head_dim})",
                )
            queries, keys = apply_rotary_pos_emb_vision(queries, keys, position_embeddings)

        attention_mask: Bool[Array, "... seq_length seq_length"] | None = None
        if cumulative_seqlens is not None:
            attention_mask = _create_mask_from_cumulative_seqlens(seq_length, cumulative_seqlens)

        attn_output = jax.nn.dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            mask=attention_mask,
        )

        attn_output_reshaped = rearrange(
            attn_output,
            "s num_heads head_dim -> s (num_heads head_dim)",
            num_heads=self.num_heads, head_dim=self.head_dim,
        )

        (output,) = vmap(self.output_proj, in_axes=0)(attn_output_reshaped)

        return output

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            qkv_proj=self.qkv_proj.export_weights(),
            output_proj=self.output_proj.export_weights(),
        )


def apply_rotary_pos_emb_vision(
    queries: Float[Array, "seq num_heads head_dim"],
    keys: Float[Array, "seq num_heads head_dim"],
    pos_emb: PositionalEmbeddings,
) -> tuple[Float[Array, "seq num_heads head_dim"], Float[Array, "seq num_heads head_dim"]]:
    cos_emb_broadcast = rearrange(pos_emb.cosines, "s d -> s 1 d")
    sin_emb_broadcast = rearrange(pos_emb.sines, "s d -> s 1 d")

    queries_rotated = pos_emb.rotate_half(queries)
    keys_rotated = pos_emb.rotate_half(keys)

    q_embed = (queries * cos_emb_broadcast) + (queries_rotated * sin_emb_broadcast)
    k_embed = (keys * cos_emb_broadcast) + (keys_rotated * sin_emb_broadcast)
    return q_embed, k_embed

def _create_mask_from_cumulative_seqlens(
    seq_length: int,
    cumulative_seqlens: Int[Array, "num_segments_plus_1"],
    dtype: jnp.dtype = jnp.bool_,
) -> Bool[Array, "1 seq_length seq_length"]:
    mask = jnp.zeros((1, seq_length, seq_length), dtype=dtype)
    num_segments = cumulative_seqlens.shape[0] - 1

    if num_segments < 0:
        raise ValueError("cumulative_seqlens must have at least one element.")

    if seq_length == 0:
        return mask

    row_indices_flat = jnp.arange(seq_length)
    col_indices_flat = jnp.arange(seq_length)

    row_indices = rearrange(row_indices_flat, "s -> 1 s 1")
    col_indices = rearrange(col_indices_flat, "t -> 1 1 t")

    row_indices = repeat(row_indices_flat, "s -> 1 s t", t=seq_length)
    col_indices = repeat(col_indices_flat, "t -> 1 s t", s=seq_length)

    def create_segment_mask(segment_idx, seqlens_array):
        start_idx = seqlens_array[segment_idx]
        end_idx = seqlens_array[segment_idx + 1]

        segment_mask = (row_indices >= start_idx) & (row_indices < end_idx) & \
                       (col_indices >= start_idx) & (col_indices < end_idx)

        return segment_mask

    result_mask = jnp.zeros((1, seq_length, seq_length), dtype=dtype)

    for i in range(num_segments):
        segment_mask = create_segment_mask(i, cumulative_seqlens)
        result_mask = result_mask | segment_mask

    return result_mask
