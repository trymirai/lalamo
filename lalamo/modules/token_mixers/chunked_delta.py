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


def _single_head(
    queries: Float[Array, "chunk_size key_channels"],
    keys: Float[Array, "chunk_size key_channels"],
    values: Float[Array, "chunk_size value_channels"],
    decay_factor: Float[Array, " chunk_size"],
    beta: Float[Array, " chunk_size"],
    ancestor_matrix: Bool[Array, "chunk_size chunk_size"] | None,
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

    if ancestor_matrix is None:
        cumulative_decay = jnp.cumsum(decay_factor.astype(jnp.float32), axis=0)
        inclusive_matrix = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool))
    else:
        cumulative_decay = ancestor_matrix.astype(jnp.float32) @ decay_factor.astype(jnp.float32)
        inclusive_matrix = ancestor_matrix
    strict_matrix = inclusive_matrix & ~jnp.eye(chunk_size, dtype=jnp.bool)

    *_, final_decay = cumulative_decay
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

    end_state_weights = jnp.exp(final_decay - cumulative_decay)
    with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
        end_state = update_values.T @ (keys * end_state_weights[:, None])

    prop_matrix = eye_chunk + jnp.where(strict_matrix, beta[:, None] * key_dot_key, 0.0)
    prop_updates = jax.scipy.linalg.solve_triangular(prop_matrix, beta[:, None] * keys, lower=True, unit_diagonal=True)

    correction = queries - jnp.where(inclusive_matrix, query_dot_key, 0.0) @ prop_updates
    correction = jnp.exp(cumulative_decay)[:, None] * correction

    end_prop = jnp.exp(final_decay) * (jnp.eye(key_channels, dtype=jnp.float32) - prop_updates.T @ keys)
    return chunk_outputs, correction, end_state, end_prop, update_values, prop_updates, cumulative_decay


def chunk_delta_forward(
    queries: Float[Array, "num_chunks chunk_size heads key_channels"],
    keys: Float[Array, "num_chunks chunk_size heads key_channels"],
    values: Float[Array, "num_chunks chunk_size heads value_channels"],
    decay_factor: Float[Array, "num_chunks chunk_size heads"],
    beta: Float[Array, "num_chunks chunk_size heads"],
    ancestor_matrix: Bool[Array, "chunk_size chunk_size"] | None = None,
) -> ChunkKernelResult:
    chunk_outputs, correction_vecs, end_state, end_prop, update_values, prop_updates, cumulative_decay = jax.vmap(
        jax.vmap(_single_head, in_axes=(1, 1, 1, 1, 1, None)),
        in_axes=(0, 0, 0, 0, 0, None),
    )(queries, keys, values, decay_factor, beta, ancestor_matrix)
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
