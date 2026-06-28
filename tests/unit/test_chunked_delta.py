from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from lalamo.modules.token_mixers.chunked_delta import chunk_delta_forward
from tests.common import assert_close


@pytest.fixture
def enable_x64() -> Iterator[None]:
    x64_was_enabled = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", val=True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", val=x64_was_enabled)


def _reference_single_head(
    queries: Array,
    keys: Array,
    values: Array,
    decay_factor: Array,
    beta: Array,
) -> tuple[Array, Array, Array, Array]:
    key_channels = keys.shape[-1]
    value_channels = values.shape[-1]

    def step(
        carry: tuple[Array, Array], inputs: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[
        tuple[Array, Array],
        tuple[Array, Array],
    ]:
        state, prop = carry
        query, key, value, log_decay, token_beta = inputs
        decay = jnp.exp(log_decay)

        decayed_state = state * decay
        state_dot_key = jnp.sum(decayed_state * key[None, :], axis=-1)
        value_update = (value - state_dot_key) * token_beta
        new_state = decayed_state + value_update[:, None] * key[None, :]

        decayed_prop = prop * decay
        prop_dot_key = decayed_prop @ key
        new_prop = decayed_prop - token_beta * prop_dot_key[:, None] * key[None, :]

        return (new_state, new_prop), (new_state @ query, new_prop @ query)

    state = jnp.zeros((value_channels, key_channels), dtype=jnp.float64)
    prop = jnp.eye(key_channels, dtype=jnp.float64)
    (end_state, end_prop), (outputs, correction_vecs) = jax.lax.scan(
        step,
        (state, prop),
        (queries, keys, values, decay_factor, beta),
    )
    return outputs, correction_vecs, end_state, end_prop


def _reference(
    queries: Array,
    keys: Array,
    values: Array,
    decay_factor: Array,
    beta: Array,
) -> tuple[Array, Array, Array, Array]:
    def per_chunk(
        chunk_queries: Array,
        chunk_keys: Array,
        chunk_values: Array,
        chunk_decay: Array,
        chunk_beta: Array,
    ) -> tuple[Array, Array, Array, Array]:
        return jax.vmap(_reference_single_head, in_axes=(1, 1, 1, 1, 1))(
            chunk_queries,
            chunk_keys,
            chunk_values,
            chunk_decay,
            chunk_beta,
        )

    queries, keys, values, decay_factor, beta = (
        array.astype(jnp.float64) for array in (queries, keys, values, decay_factor, beta)
    )
    outputs, correction_vecs, end_state, end_prop = jax.vmap(per_chunk, in_axes=(0, 0, 0, 0, 0))(
        queries,
        keys,
        values,
        decay_factor,
        beta,
    )
    return (
        jnp.transpose(outputs, (0, 2, 1, 3)),
        jnp.transpose(correction_vecs, (0, 2, 1, 3)),
        end_state,
        end_prop,
    )


def _make_inputs(
    num_chunks: int,
    chunk_size: int,
    num_heads: int,
    key_channels: int,
    value_channels: int,
) -> tuple[Array, Array, Array, Array, Array]:
    query_key, key_key, value_key, beta_key, decay_key = jax.random.split(jax.random.key(0), 5)
    vector_shape = (num_chunks, chunk_size, num_heads, key_channels)
    queries = jax.random.normal(query_key, vector_shape, dtype=jnp.float64)
    keys = jax.random.normal(key_key, vector_shape, dtype=jnp.float64)
    keys = keys / jnp.linalg.norm(keys, axis=-1, keepdims=True)
    values = jax.random.normal(value_key, (num_chunks, chunk_size, num_heads, value_channels), dtype=jnp.float64)
    beta = jax.nn.sigmoid(jax.random.normal(beta_key, (num_chunks, chunk_size, num_heads), dtype=jnp.float64))
    decay_factor = -jax.nn.softplus(
        jax.random.normal(decay_key, (num_chunks, chunk_size, num_heads), dtype=jnp.float64)
    )
    return queries, keys, values, decay_factor, beta


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=result, reference=reference.astype(jnp.float32), rtol=1e-4, atol=1e-5)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize("chunk_size", [32, 64, 128])
def test_chunk_delta_matches_reference(chunk_size: int) -> None:
    inputs = _make_inputs(
        num_chunks=3,
        chunk_size=chunk_size,
        num_heads=4,
        key_channels=96,
        value_channels=128,
    )

    expected_outputs, expected_correction_vecs, expected_end_state, expected_end_prop = _reference(*inputs)
    result = chunk_delta_forward(*inputs)

    _assert_close(result.chunk_outputs, expected_outputs)
    _assert_close(result.correction_vecs, expected_correction_vecs)
    _assert_close(result.end_state, expected_end_state)
    _assert_close(result.end_prop, expected_end_prop)
