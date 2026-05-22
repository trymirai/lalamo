from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, Key

__all__ = [
    "RankedTokens",
    "fold_gumbel_key",
    "rank_token_logits",
    "token_gumbels",
]


class RankedTokens(NamedTuple):
    token_ids: Int[Array, "*rows width"]
    scores: Float[Array, "*rows width"]
    candidate_indices: Int[Array, "*rows width"]
    valid: Bool[Array, "*rows width"]


def fold_gumbel_key(
    key: Key[Array, ""],
    position: Int[Array, ""],
    seed: Int[Array, ""],
) -> Key[Array, ""]:
    key = jax.random.fold_in(key, position.astype(jnp.int32))
    return jax.random.fold_in(key, seed.astype(jnp.int32))


def token_gumbels(
    keys: Key[Array, "*rows"],
    positions: Int[Array, "*rows"],
    seeds: Int[Array, "*rows"],
    token_ids: Int[Array, "*rows candidates"],
    dtype: DTypeLike = jnp.float32,
) -> Float[Array, "*rows candidates"]:
    dtype = jnp.dtype(dtype)
    key0 = keys[..., 0].astype(jnp.uint32)
    key1 = keys[..., 1].astype(jnp.uint32)
    position_bits = positions.astype(jnp.uint32)
    seed_bits = seeds.astype(jnp.uint32)
    bits = token_ids.astype(jnp.uint32)
    bits = bits ^ (key0[..., None] + jnp.asarray(0x9E3779B9, dtype=jnp.uint32))
    bits = bits ^ (key1[..., None] + jnp.asarray(0xBB67AE85, dtype=jnp.uint32))
    bits = bits ^ (position_bits[..., None] * jnp.asarray(0x85EBCA6B, dtype=jnp.uint32))
    bits = bits ^ (seed_bits[..., None] * jnp.asarray(0xC2B2AE35, dtype=jnp.uint32))
    bits = bits ^ (bits >> jnp.asarray(16, dtype=jnp.uint32))
    bits = bits * jnp.asarray(0x7FEB352D, dtype=jnp.uint32)
    bits = bits ^ (bits >> jnp.asarray(15, dtype=jnp.uint32))
    bits = bits * jnp.asarray(0x846CA68B, dtype=jnp.uint32)
    bits = bits ^ (bits >> jnp.asarray(16, dtype=jnp.uint32))
    uniform_bits = (bits >> jnp.asarray(8, dtype=jnp.uint32)).astype(dtype)
    uniform = (uniform_bits + jnp.asarray(0.5, dtype=dtype)) * jnp.asarray(1.0 / 16777216.0, dtype=dtype)
    return -jnp.log(-jnp.log(uniform))


def rank_token_logits(
    token_ids: Int[Array, "*rows candidates"],
    processed_logits: Float[Array, "*rows candidates"],
    candidate_mask: Bool[Array, "*rows candidates"],
    keys: Key[Array, "*rows"],
    positions: Int[Array, "*rows"],
    seeds: Int[Array, "*rows"],
    temperature: Float[Array, "*rows"] | None,
    width: int,
) -> RankedTokens:
    if token_ids.shape != processed_logits.shape:
        raise ValueError("token_ids and processed_logits must have the same shape.")
    if candidate_mask.shape != token_ids.shape:
        raise ValueError("candidate_mask must match token_ids shape.")
    if width < 1 or width > token_ids.shape[-1]:
        raise ValueError("width must be between 1 and candidate count.")

    processed_logits = processed_logits.astype(jnp.float32)
    if temperature is None:
        scores = processed_logits + token_gumbels(keys, positions, seeds, token_ids, processed_logits.dtype)
    else:
        row_shape = token_ids.shape[:-1]
        temperature = jnp.broadcast_to(jnp.asarray(temperature, dtype=processed_logits.dtype), row_shape)
        scores = jax.lax.cond(
            jnp.all(temperature < 1e-5),
            lambda _: processed_logits,
            lambda _: jnp.where(
                temperature[..., None] < 1e-5,
                processed_logits,
                processed_logits + token_gumbels(keys, positions, seeds, token_ids, processed_logits.dtype),
            ),
            operand=None,
        )

    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    ranked_scores, candidate_indices = jax.lax.top_k(scores, width)
    return RankedTokens(
        token_ids=jnp.take_along_axis(token_ids, candidate_indices, axis=-1).astype(jnp.int32),
        scores=ranked_scores,
        candidate_indices=candidate_indices.astype(jnp.int32),
        valid=jnp.take_along_axis(candidate_mask, candidate_indices, axis=-1),
    )
