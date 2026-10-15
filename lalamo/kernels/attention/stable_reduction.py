import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jaxtyping import Array, Bool, DTypeLike, Float


def _repeat_key_values(
    keys_or_values: Float[Array, "tokens groups channels"],
    group_size: int,
) -> Float[Array, "tokens heads channels"]:
    if group_size == 1:
        return keys_or_values
    return jnp.repeat(keys_or_values, group_size, axis=1)


def stable_reduction_attention(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    *,
    bias: Float[Array, "heads dst_tokens src_tokens"] | None,
    mask: Bool[Array, "dst_tokens src_tokens"],
    scale: float | Float[Array, ""] | None,
    logit_soft_cap: float | Float[Array, ""] | None,
    tile_size: int,
    accumulation_dtype: DTypeLike | None,
) -> Float[Array, "dst_tokens heads head_channels"]:
    if tile_size < 1:
        raise ValueError("attention_tile_size must be at least 1.")

    original_dtype = queries.dtype
    accumulation_dtype = original_dtype if accumulation_dtype is None else accumulation_dtype

    _, num_heads, head_dim = queries.shape
    num_keys, num_groups, _ = keys.shape
    group_size = num_heads // num_groups

    if scale is None:
        scale = head_dim**-0.5

    if group_size > 1:
        keys = _repeat_key_values(keys, group_size)
        values = _repeat_key_values(values, group_size)

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
        scores = jax.nn.tanh(scores / logit_soft_cap) * logit_soft_cap
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
    normalizer = final_sum[-1]
    safe_normalizer = jnp.where(normalizer > 0, normalizer, 1)
    result = final_output[-1] / safe_normalizer[..., None]
    return rearrange(result, "heads queries channels -> queries heads channels").astype(original_dtype)
