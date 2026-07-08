from typing import NamedTuple

import einops
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Bool, Float

from lalamo.utils.precision import use_dot_algorithm_preset


class ChunkKernelResult(NamedTuple):
    chunk_outputs: Float[Array, "num_chunks chunk_size heads value_channels"]
    correction_vecs: Float[Array, "num_chunks chunk_size heads key_channels"]
    end_state: Float[Array, "num_chunks heads value_channels key_channels"]
    end_prop: Float[Array, "num_chunks heads key_channels key_channels"]
    update_values: Float[Array, "num_chunks chunk_size heads value_channels"]
    prop_updates: Float[Array, "num_chunks chunk_size heads key_channels"]
    cumulative_decay: Float[Array, "num_chunks chunk_size heads"]


class TreeVerifyResult(NamedTuple):
    outputs: Float[Array, "tokens heads value_channels"]
    correction_vecs: Float[Array, "tokens heads key_channels"]
    update_values: Float[Array, "tokens heads value_channels"]
    prop_updates: Float[Array, "tokens heads key_channels"]
    cumulative_decay: Float[Array, "tokens heads"]


def _intra_chunk_core(
    queries: Float[Array, "chunk_size key_channels"],
    keys: Float[Array, "chunk_size key_channels"],
    values: Float[Array, "chunk_size value_channels"],
    beta: Float[Array, " chunk_size"],
    cumulative_decay: Float[Array, " chunk_size"],
    inclusive_matrix: Bool[Array, "chunk_size chunk_size"],
) -> tuple[
    Float[Array, "chunk_size value_channels"],
    Float[Array, "chunk_size key_channels"],
    Float[Array, "chunk_size value_channels"],
    Float[Array, "chunk_size key_channels"],
]:
    chunk_size, _ = keys.shape
    strict_matrix = inclusive_matrix & ~jnp.eye(chunk_size, dtype=jnp.bool)
    pair_decay = cumulative_decay[:, None] - cumulative_decay[None, :]

    # Mask before exp so unused off-path entries cannot overflow.
    lower_decay = jnp.exp(jnp.where(strict_matrix, pair_decay, 0.0))
    key_dot_key = keys @ keys.T
    eye_chunk = jnp.eye(chunk_size, dtype=jnp.float32)
    state_matrix = eye_chunk + jnp.where(strict_matrix, beta[:, None] * lower_decay * key_dot_key, 0.0)

    update_values = jax.scipy.linalg.solve_triangular(
        state_matrix, beta[:, None] * values, lower=True, unit_diagonal=True
    )

    inclusive_decay = jnp.exp(jnp.where(inclusive_matrix, pair_decay, 0.0))
    query_dot_key = queries @ keys.T
    attention = jnp.where(inclusive_matrix, inclusive_decay * query_dot_key, 0.0)
    chunk_outputs = attention @ update_values

    prop_matrix = eye_chunk + jnp.where(strict_matrix, beta[:, None] * key_dot_key, 0.0)
    prop_updates = jax.scipy.linalg.solve_triangular(prop_matrix, beta[:, None] * keys, lower=True, unit_diagonal=True)

    correction = queries - jnp.where(inclusive_matrix, query_dot_key, 0.0) @ prop_updates
    correction = jnp.exp(cumulative_decay)[:, None] * correction
    return chunk_outputs, correction, update_values, prop_updates


def _single_head(
    queries: Float[Array, "chunk_size key_channels"],
    keys: Float[Array, "chunk_size key_channels"],
    values: Float[Array, "chunk_size value_channels"],
    decay_factor: Float[Array, " chunk_size"],
    beta: Float[Array, " chunk_size"],
) -> tuple[
    Float[Array, "chunk_size value_channels"],
    Float[Array, "chunk_size key_channels"],
    Float[Array, "value_channels key_channels"],
    Float[Array, "key_channels key_channels"],
    Float[Array, "chunk_size value_channels"],
    Float[Array, "chunk_size key_channels"],
    Float[Array, " chunk_size"],
]:
    chunk_size, key_channels = keys.shape
    beta = beta.astype(jnp.float32)
    queries = queries.astype(jnp.float32)
    keys = keys.astype(jnp.float32)
    values = values.astype(jnp.float32)

    cumulative_decay = jnp.cumsum(decay_factor.astype(jnp.float32), axis=0)
    inclusive_matrix = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool))
    chunk_outputs, correction, update_values, prop_updates = _intra_chunk_core(
        queries, keys, values, beta, cumulative_decay, inclusive_matrix
    )

    *_, final_decay = cumulative_decay
    end_state_weights = jnp.exp(final_decay - cumulative_decay)
    with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
        end_state = update_values.T @ (keys * end_state_weights[:, None])

    end_prop = jnp.exp(final_decay) * (jnp.eye(key_channels, dtype=jnp.float32) - prop_updates.T @ keys)
    return chunk_outputs, correction, end_state, end_prop, update_values, prop_updates, cumulative_decay


def _single_head_tree(
    queries: Float[Array, "tokens key_channels"],
    keys: Float[Array, "tokens key_channels"],
    values: Float[Array, "tokens value_channels"],
    decay_factor: Float[Array, " tokens"],
    beta: Float[Array, " tokens"],
    ancestor_matrix: Bool[Array, "tokens tokens"],
) -> tuple[
    Float[Array, "tokens value_channels"],
    Float[Array, "tokens key_channels"],
    Float[Array, "tokens value_channels"],
    Float[Array, "tokens key_channels"],
    Float[Array, " tokens"],
]:
    beta = beta.astype(jnp.float32)
    queries = queries.astype(jnp.float32)
    keys = keys.astype(jnp.float32)
    values = values.astype(jnp.float32)

    # Decay sums feed exp(); keep them out of tf32 regardless of the backend default.
    with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
        cumulative_decay = ancestor_matrix.astype(jnp.float32) @ decay_factor.astype(jnp.float32)
    chunk_outputs, correction, update_values, prop_updates = _intra_chunk_core(
        queries, keys, values, beta, cumulative_decay, ancestor_matrix
    )
    return chunk_outputs, correction, update_values, prop_updates, cumulative_decay


def chunk_delta_forward(
    queries: Float[Array, "num_chunks chunk_size heads key_channels"],
    keys: Float[Array, "num_chunks chunk_size heads key_channels"],
    values: Float[Array, "num_chunks chunk_size heads value_channels"],
    decay_factor: Float[Array, "num_chunks chunk_size heads"],
    beta: Float[Array, "num_chunks chunk_size heads"],
) -> ChunkKernelResult:
    chunk_outputs, correction_vecs, end_state, end_prop, update_values, prop_updates, cumulative_decay = jax.vmap(
        jax.vmap(_single_head, in_axes=(1, 1, 1, 1, 1)),
        in_axes=(0, 0, 0, 0, 0),
    )(queries, keys, values, decay_factor, beta)
    return ChunkKernelResult(
        einops.rearrange(
            chunk_outputs,
            "num_chunks heads chunk_size value_channels -> num_chunks chunk_size heads value_channels",
        ),
        einops.rearrange(
            correction_vecs,
            "num_chunks heads chunk_size key_channels -> num_chunks chunk_size heads key_channels",
        ),
        end_state,
        end_prop,
        einops.rearrange(
            update_values,
            "num_chunks heads chunk_size value_channels -> num_chunks chunk_size heads value_channels",
        ),
        einops.rearrange(
            prop_updates,
            "num_chunks heads chunk_size key_channels -> num_chunks chunk_size heads key_channels",
        ),
        einops.rearrange(cumulative_decay, "num_chunks heads chunk_size -> num_chunks chunk_size heads"),
    )


def tree_delta_verify(
    queries: Float[Array, "tokens heads key_channels"],
    keys: Float[Array, "tokens heads key_channels"],
    values: Float[Array, "tokens heads value_channels"],
    decay_factor: Float[Array, "tokens heads"],
    beta: Float[Array, "tokens heads"],
    ancestor_matrix: Bool[Array, "tokens tokens"],
) -> TreeVerifyResult:
    # Verification is read-only under lag-folding: no end state or propagator is needed,
    # and the tree is always a single chunk.
    outputs, correction_vecs, update_values, prop_updates, cumulative_decay = jax.vmap(
        _single_head_tree,
        in_axes=(1, 1, 1, 1, 1, None),
    )(queries, keys, values, decay_factor, beta, ancestor_matrix)
    return TreeVerifyResult(
        einops.rearrange(outputs, "heads tokens value_channels -> tokens heads value_channels"),
        einops.rearrange(correction_vecs, "heads tokens key_channels -> tokens heads key_channels"),
        einops.rearrange(update_values, "heads tokens value_channels -> tokens heads value_channels"),
        einops.rearrange(prop_updates, "heads tokens key_channels -> tokens heads key_channels"),
        einops.rearrange(cumulative_decay, "heads tokens -> tokens heads"),
    )
