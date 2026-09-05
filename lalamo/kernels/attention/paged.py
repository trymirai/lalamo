import jax
from jax.experimental.pallas.ops.gpu.decode_attention import gqa
from jax.experimental.pallas.ops.gpu.paged_attention import paged_attention
from jaxtyping import Array, Float, Int

__all__ = ["paged_decode_attention", "windowed_decode_attention"]


def paged_decode_attention(
    queries: Float[Array, "batch heads head_channels"],
    key_pages: Float[Array, "groups total_pages page_size head_channels"],
    value_pages: Float[Array, "groups total_pages page_size head_channels"],
    block_tables: Int[Array, "batch pages_per_sequence"],
    lengths: Int[Array, " batch"],
    *,
    scale: float,
    logit_soft_cap: float | None,
) -> Float[Array, "batch heads head_channels"]:
    return paged_attention(
        queries * scale,
        key_pages,
        value_pages,
        block_tables,
        lengths,
        pages_per_compute_block=1,
        k_splits=min(block_tables.shape[1], 16),
        num_stages=1,
        attn_logits_soft_cap=logit_soft_cap,
    )


def windowed_decode_attention(
    queries: Float[Array, "batch heads head_channels"],
    keys: Float[Array, "batch tokens groups head_channels"],
    values: Float[Array, "batch tokens groups head_channels"],
    start_indices: Int[Array, " batch"],
    end_indices: Int[Array, " batch"],
    *,
    scale: float,
) -> Float[Array, "batch heads head_channels"]:
    _, tokens, _, _ = keys.shape
    with jax.numpy_dtype_promotion("standard"):
        return gqa(
            queries,
            keys,
            values,
            start_indices,
            end_indices,
            sm_scale=scale,
            k_splits=min(tokens // 32, 16),
        ).astype(queries.dtype)
