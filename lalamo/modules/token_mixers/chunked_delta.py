from typing import NamedTuple

import einops
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Float

from lalamo.utils.precision import use_dot_algorithm_preset


class ChunkKernelResult(NamedTuple):
    chunk_outputs: Float[Array, "num_chunks chunk_size heads value_channels"]
    correction_vecs: Float[Array, "num_chunks chunk_size heads key_channels"]
    end_state: Float[Array, "num_chunks heads value_channels key_channels"]
    end_prop: Float[Array, "num_chunks heads key_channels key_channels"]


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
]:
    chunk_size, key_channels = keys.shape
    beta = beta.astype(jnp.float32)
    queries = queries.astype(jnp.float32)
    keys = keys.astype(jnp.float32)
    values = values.astype(jnp.float32)

    cumulative_decay = jnp.cumsum(decay_factor.astype(jnp.float32), axis=0)
    *_, final_decay = cumulative_decay
    pair_decay = cumulative_decay[:, None] - cumulative_decay[None, :]

    # Mask before exp so unused upper-triangle entries cannot overflow.
    lower_decay = jnp.exp(jnp.tril(pair_decay, -1))
    key_dot_key = keys @ keys.T
    eye_chunk = jnp.eye(chunk_size, dtype=jnp.float32)
    state_matrix = eye_chunk + jnp.tril(beta[:, None] * lower_decay * key_dot_key, -1)

    update_values = jax.scipy.linalg.solve_triangular(
        state_matrix, beta[:, None] * values, lower=True, unit_diagonal=True
    )

    inclusive_decay = jnp.exp(jnp.tril(pair_decay, 0))
    query_dot_key = queries @ keys.T
    attention = jnp.tril(inclusive_decay * query_dot_key, 0)
    chunk_outputs = attention @ update_values

    end_state_weights = jnp.exp(final_decay - cumulative_decay)
    with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
        end_state = update_values.T @ (keys * end_state_weights[:, None])

    prop_matrix = eye_chunk + jnp.tril(beta[:, None] * key_dot_key, -1)
    prop_updates = jax.scipy.linalg.solve_triangular(prop_matrix, beta[:, None] * keys, lower=True, unit_diagonal=True)

    correction = queries - jnp.tril(query_dot_key, 0) @ prop_updates
    correction = jnp.exp(cumulative_decay)[:, None] * correction

    end_prop = jnp.exp(final_decay) * (jnp.eye(key_channels, dtype=jnp.float32) - prop_updates.T @ keys)
    return chunk_outputs, correction, end_state, end_prop


def chunk_delta_forward(
    queries: Float[Array, "num_chunks chunk_size heads key_channels"],
    keys: Float[Array, "num_chunks chunk_size heads key_channels"],
    values: Float[Array, "num_chunks chunk_size heads value_channels"],
    decay_factor: Float[Array, "num_chunks chunk_size heads"],
    beta: Float[Array, "num_chunks chunk_size heads"],
) -> ChunkKernelResult:
    chunk_outputs, correction_vecs, end_state, end_prop = jax.vmap(
        jax.vmap(_single_head, in_axes=1),
        in_axes=0,
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
    )
