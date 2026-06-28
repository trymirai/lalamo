"""Closed-form DeltaNet intra-chunk recurrence."""

from __future__ import annotations

from typing import NamedTuple

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class ChunkKernelResult(NamedTuple):
    chunk_outputs: Float[Array, "num_chunks chunk_size heads value_channels"]
    correction_vecs: Float[Array, "num_chunks chunk_size heads key_channels"]
    end_state: Float[Array, "num_chunks heads value_channels key_channels"]
    end_prop: Float[Array, "num_chunks heads key_channels key_channels"]


def _matmul(
    left: Array,
    right: Array,
    precision: jax.lax.Precision | None = None,
) -> Array:
    return jnp.matmul(
        left,
        right,
        preferred_element_type=jnp.float32,
        precision=precision,
    )


def _solve_unit_lower(matrix: Array, rhs: Array) -> Array:
    return jax.scipy.linalg.solve_triangular(
        matrix,
        rhs,
        lower=True,
        unit_diagonal=True,
    )


def _single_head(
    queries: Float[Array, "chunk_size key_channels"],
    keys: Float[Array, "chunk_size key_channels"],
    values: Float[Array, "chunk_size value_channels"],
    decay_factor: Array,
    beta: Array,
) -> tuple[
    Float[Array, "chunk_size value_channels"],
    Float[Array, "chunk_size key_channels"],
    Float[Array, "value_channels key_channels"],
    Float[Array, "key_channels key_channels"],
]:
    """Closed-form intra-chunk pass for one head: (outputs, correction_vecs, end_state, end_prop)."""
    chunk_size, key_channels = keys.shape
    beta = beta.astype(jnp.float32)
    queries = queries.astype(jnp.float32)
    keys = keys.astype(jnp.float32)
    values = values.astype(jnp.float32)

    cumulative_decay = jnp.cumsum(decay_factor.astype(jnp.float32), axis=0)
    final_decay = cumulative_decay[-1]
    pair_decay = cumulative_decay[:, None] - cumulative_decay[None, :]

    # Mask before exp so unused upper-triangle entries cannot overflow.
    lower_decay = jnp.exp(jnp.tril(pair_decay, -1))
    key_dot_key = _matmul(keys, keys.T)
    eye_chunk = jnp.eye(chunk_size, dtype=jnp.float32)
    state_matrix = eye_chunk + jnp.tril(beta[:, None] * lower_decay * key_dot_key, -1)

    update_values = _solve_unit_lower(state_matrix, beta[:, None] * values)

    inclusive_decay = jnp.exp(jnp.tril(pair_decay, 0))
    query_dot_key = _matmul(queries, keys.T)
    attention = jnp.tril(inclusive_decay * query_dot_key, 0)
    chunk_outputs = _matmul(attention, update_values)

    end_state_weights = jnp.exp(final_decay - cumulative_decay)
    end_state = _matmul(
        update_values.T,
        keys * end_state_weights[:, None],
        precision=jax.lax.Precision.HIGHEST,
    )

    prop_matrix = eye_chunk + jnp.tril(beta[:, None] * key_dot_key, -1)
    prop_updates = _solve_unit_lower(prop_matrix, beta[:, None] * keys)

    key_dot_query = _matmul(keys, queries.T)
    past_key_dot_query = jnp.triu(key_dot_query, 0)
    correction = queries - _matmul(past_key_dot_query.T, prop_updates)
    correction = jnp.exp(cumulative_decay)[:, None] * correction

    end_prop = jnp.exp(final_decay) * (jnp.eye(key_channels, dtype=jnp.float32) - _matmul(prop_updates.T, keys))
    return chunk_outputs, correction, end_state, end_prop


def chunk_delta_forward(
    queries: Float[Array, "num_chunks chunk_size heads key_channels"],
    keys: Float[Array, "num_chunks chunk_size heads key_channels"],
    values: Float[Array, "num_chunks chunk_size heads value_channels"],
    decay_factor: Float[Array, "num_chunks chunk_size heads"],
    beta: Float[Array, "num_chunks chunk_size heads"],
) -> ChunkKernelResult:
    def per_chunk(
        chunk_inputs: tuple[
            Float[Array, "chunk_size heads key_channels"],
            Float[Array, "chunk_size heads key_channels"],
            Float[Array, "chunk_size heads value_channels"],
            Float[Array, "chunk_size heads"],
            Float[Array, "chunk_size heads"],
        ],
    ) -> tuple[Array, Array, Array, Array]:
        chunk_queries, chunk_keys, chunk_values, chunk_decay, chunk_beta = chunk_inputs
        return jax.vmap(_single_head, in_axes=(1, 1, 1, 1, 1))(
            chunk_queries,
            chunk_keys,
            chunk_values,
            chunk_decay,
            chunk_beta,
        )

    chunk_outputs, correction_vecs, end_state, end_prop = jax.vmap(per_chunk, in_axes=0)(
        (queries, keys, values, decay_factor, beta),
    )
    return ChunkKernelResult(
        einops.rearrange(
            chunk_outputs,
            "num_chunks heads chunk_size value_channels -> num_chunks chunk_size heads value_channels",
        ).astype(jnp.float32),
        einops.rearrange(
            correction_vecs,
            "num_chunks heads chunk_size key_channels -> num_chunks chunk_size heads key_channels",
        ).astype(jnp.float32),
        end_state.astype(jnp.float32),
        end_prop.astype(jnp.float32),
    )
