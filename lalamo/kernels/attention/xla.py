from typing import cast

import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jaxtyping import Array, Bool, Float


def xla_attention(
    queries: Float[Array, "dst_tokens heads head_dim"],
    keys: Float[Array, "src_tokens key_value_heads head_dim"],
    values: Float[Array, "src_tokens key_value_heads head_dim"],
    bias: Float[Array, "heads dst_tokens src_tokens"] | None,
    mask: Bool[Array, "dst_tokens src_tokens"],
    scale: float | Float[Array, ""] | None,
    logit_soft_cap: float | None,
) -> Float[Array, "dst_tokens heads head_dim"]:
    original_dtype = queries.dtype
    if logit_soft_cap is not None:
        queries = queries.astype(jnp.float32)
        keys = keys.astype(jnp.float32)
        values = values.astype(jnp.float32)
        if bias is not None:
            bias = bias.astype(jnp.float32)

        _, num_heads, head_dim = queries.shape
        _, num_key_value_heads, _ = keys.shape
        heads_per_key_value_head = num_heads // num_key_value_heads
        if heads_per_key_value_head > 1:
            keys = jnp.repeat(keys, heads_per_key_value_head, axis=1)
            values = jnp.repeat(values, heads_per_key_value_head, axis=1)
        queries_head_first = rearrange(
            queries,
            "dst_tokens heads channels -> heads dst_tokens channels",
        )
        keys_head_first = rearrange(
            keys,
            "src_tokens heads channels -> heads src_tokens channels",
        )
        attention_logits = einsum(
            queries_head_first,
            keys_head_first,
            "heads dst_tokens channels, heads src_tokens channels -> heads dst_tokens src_tokens",
        )
        attention_scale = head_dim**-0.5 if scale is None else scale
        attention_logits = attention_logits * attention_scale
        attention_logits = jax.nn.tanh(attention_logits / logit_soft_cap) * logit_soft_cap
        if bias is not None:
            attention_logits = attention_logits + bias
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

    with jax.numpy_dtype_promotion("standard"):
        return jax.nn.dot_product_attention(
            queries,
            keys,
            values,
            bias=bias,
            mask=mask,
            scale=cast("float | None", scale),
            implementation="xla",
        ).astype(original_dtype)
