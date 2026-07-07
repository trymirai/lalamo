from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest
from einops import einsum
from jaxtyping import Array

from lalamo.modules.token_mixers.chunked_delta import ChunkKernelResult, chunk_delta_forward
from lalamo.modules.token_mixers.kv_cache import tree_ancestor_mask
from lalamo.modules.token_mixers.ssm_state import fold_lag_factors
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
    queries, keys, values, decay_factor, beta = (
        array.astype(jnp.float64) for array in (queries, keys, values, decay_factor, beta)
    )
    outputs, correction_vecs, end_state, end_prop = jax.vmap(
        jax.vmap(_reference_single_head, in_axes=1),
        in_axes=0,
    )(queries, keys, values, decay_factor, beta)
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

    for actual, expected in (
        (result.chunk_outputs, expected_outputs),
        (result.correction_vecs, expected_correction_vecs),
        (result.end_state, expected_end_state),
        (result.end_prop, expected_end_prop),
    ):
        assert_close(result=actual, reference=expected.astype(jnp.float32), rtol=1e-4, atol=1e-5)


CHAIN_PARENTS = [-1, 0, 1, 2, 3, 4, 5]
TREE_PARENTS = [-1, 0, 0, 1, 2, 2, 4]
NUM_NODES = 7
NUM_HEADS = 2
KEY_CHANNELS = 8
VALUE_CHANNELS = 6
CONV_HISTORY = 3
CONV_CHANNELS = 5


def root_path(parent_indices: list[int], node: int) -> list[int]:
    path = []
    cursor = node
    while cursor >= 0:
        path.append(cursor)
        cursor = parent_indices[cursor]
    return path[::-1]


def make_tree_inputs() -> tuple[Array, Array, Array, Array, Array, Array]:
    query_key, key_key, value_key, beta_key, decay_key, state_key = jax.random.split(jax.random.key(0), 6)
    vector_shape = (NUM_NODES, NUM_HEADS, KEY_CHANNELS)
    queries = jax.random.normal(query_key, vector_shape, dtype=jnp.float64)
    keys = jax.random.normal(key_key, vector_shape, dtype=jnp.float64)
    keys = keys / jnp.linalg.norm(keys, axis=-1, keepdims=True)
    values = jax.random.normal(value_key, (NUM_NODES, NUM_HEADS, VALUE_CHANNELS), dtype=jnp.float64)
    beta = jax.nn.sigmoid(jax.random.normal(beta_key, (NUM_NODES, NUM_HEADS), dtype=jnp.float64))
    decay_factor = -jax.nn.softplus(jax.random.normal(decay_key, (NUM_NODES, NUM_HEADS), dtype=jnp.float64))
    initial_state = jax.random.normal(state_key, (NUM_HEADS, VALUE_CHANNELS, KEY_CHANNELS), dtype=jnp.float64)
    return queries, keys, values, decay_factor, beta, initial_state


def tree_verify(
    queries: Array,
    keys: Array,
    values: Array,
    decay_factor: Array,
    beta: Array,
    initial_state: Array,
    parent_indices: Array,
) -> tuple[Array, ChunkKernelResult]:
    ancestor_matrix = tree_ancestor_mask(parent_indices)
    result = chunk_delta_forward(
        queries[None],
        keys[None],
        values[None],
        decay_factor[None],
        beta[None],
        ancestor_matrix,
    )
    outputs = result.chunk_outputs[0] + einsum(
        initial_state.astype(jnp.float32),
        result.correction_vecs[0],
        "heads value_channels key_channels, nodes heads key_channels -> nodes heads value_channels",
    )
    return outputs, result


def reference_path_state(
    keys: Array,
    values: Array,
    decay_factor: Array,
    beta: Array,
    initial_state: Array,
    path: list[int],
    head: int,
) -> tuple[Array, Array]:
    state = initial_state[head]
    value_update = jnp.zeros_like(values[0, head])
    for step in path:
        state = state * jnp.exp(decay_factor[step, head])
        value_update = beta[step, head] * (values[step, head] - state @ keys[step, head])
        state = state + value_update[:, None] * keys[step, head][None, :]
    return state, value_update


def reference_tree_verify(
    queries: Array,
    keys: Array,
    values: Array,
    decay_factor: Array,
    beta: Array,
    initial_state: Array,
    parent_indices: list[int],
) -> tuple[Array, Array, Array]:
    outputs, update_values, cumulative_decay = [], [], []
    for node in range(NUM_NODES):
        path = root_path(parent_indices, node)
        node_outputs, node_updates, node_decay = [], [], []
        for head in range(NUM_HEADS):
            state, value_update = reference_path_state(keys, values, decay_factor, beta, initial_state, path, head)
            node_outputs.append(state @ queries[node, head])
            node_updates.append(value_update)
            node_decay.append(decay_factor[path, head].sum())
        outputs.append(jnp.stack(node_outputs))
        update_values.append(jnp.stack(node_updates))
        cumulative_decay.append(jnp.stack(node_decay))
    return jnp.stack(outputs), jnp.stack(update_values), jnp.stack(cumulative_decay)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize("parents", [CHAIN_PARENTS, TREE_PARENTS], ids=["chain", "tree"])
def test_tree_verify_matches_recurrent_reference(parents: list[int]) -> None:
    queries, keys, values, decay_factor, beta, initial_state = make_tree_inputs()
    parent_indices = jnp.array(parents, dtype=jnp.int32)

    outputs, result = tree_verify(queries, keys, values, decay_factor, beta, initial_state, parent_indices)
    expected_outputs, _, expected_decay = reference_tree_verify(
        queries, keys, values, decay_factor, beta, initial_state, parents
    )
    _, expected_updates, _ = reference_tree_verify(
        queries, keys, values, decay_factor, beta, jnp.zeros_like(initial_state), parents
    )

    assert_close(result=outputs, reference=expected_outputs.astype(jnp.float32), rtol=1e-4, atol=1e-5)
    assert_close(result=result.update_values[0], reference=expected_updates.astype(jnp.float32), rtol=1e-4, atol=1e-5)
    assert_close(result=result.cumulative_decay[0], reference=expected_decay.astype(jnp.float32), rtol=1e-4, atol=1e-5)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize("leaf", range(NUM_NODES))
def test_fold_lag_factors_matches_recurrent_replay(leaf: int) -> None:
    queries, keys, values, decay_factor, beta, initial_state = make_tree_inputs()
    parent_indices = jnp.array(TREE_PARENTS, dtype=jnp.int32)
    _, verify_result = tree_verify(queries, keys, values, decay_factor, beta, initial_state, parent_indices)

    path = root_path(TREE_PARENTS, leaf)
    accepted_node_indices = jnp.full((NUM_NODES,), -1, dtype=jnp.int32)
    accepted_node_indices = accepted_node_indices.at[: len(path)].set(jnp.array(path, dtype=jnp.int32))
    conv_key, window_key = jax.random.split(jax.random.key(1))
    conv_state = jax.random.normal(conv_key, (CONV_HISTORY, CONV_CHANNELS), dtype=jnp.float32)
    conv_windows = jax.random.normal(window_key, (NUM_NODES, CONV_HISTORY, CONV_CHANNELS), dtype=jnp.float32)

    new_conv_state, new_ssm_state = fold_lag_factors(
        conv_state,
        initial_state.astype(jnp.float32),
        keys.astype(jnp.float32),
        verify_result.update_values[0],
        verify_result.prop_updates[0],
        verify_result.cumulative_decay[0],
        conv_windows,
        accepted_node_indices,
        jnp.asarray(len(path), dtype=jnp.int32),
    )

    expected_state = jnp.stack(
        [
            reference_path_state(keys, values, decay_factor, beta, initial_state, path, head)[0]
            for head in range(NUM_HEADS)
        ]
    )
    assert_close(result=new_ssm_state, reference=expected_state.astype(jnp.float32), rtol=1e-4, atol=1e-5)
    assert jnp.array_equal(new_conv_state, conv_windows[leaf])


@pytest.mark.usefixtures("enable_x64")
def test_fold_lag_factors_without_accepted_nodes_keeps_committed_state() -> None:
    queries, keys, values, decay_factor, beta, initial_state = make_tree_inputs()
    parent_indices = jnp.array(TREE_PARENTS, dtype=jnp.int32)
    _, verify_result = tree_verify(queries, keys, values, decay_factor, beta, initial_state, parent_indices)

    conv_key, window_key = jax.random.split(jax.random.key(2))
    conv_state = jax.random.normal(conv_key, (CONV_HISTORY, CONV_CHANNELS), dtype=jnp.float32)
    conv_windows = jax.random.normal(window_key, (NUM_NODES, CONV_HISTORY, CONV_CHANNELS), dtype=jnp.float32)
    committed_state = initial_state.astype(jnp.float32)

    new_conv_state, new_ssm_state = fold_lag_factors(
        conv_state,
        committed_state,
        keys.astype(jnp.float32),
        verify_result.update_values[0],
        verify_result.prop_updates[0],
        verify_result.cumulative_decay[0],
        conv_windows,
        jnp.full((NUM_NODES,), -1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert jnp.array_equal(new_conv_state, conv_state)
    assert jnp.array_equal(new_ssm_state, committed_state)


@pytest.mark.usefixtures("enable_x64")
def test_tree_verify_sibling_isolation() -> None:
    queries, keys, values, decay_factor, beta, initial_state = make_tree_inputs()
    parent_indices = jnp.array(TREE_PARENTS, dtype=jnp.int32)

    outputs, _ = tree_verify(queries, keys, values, decay_factor, beta, initial_state, parent_indices)
    perturbed_values = values.at[2].add(1.0)
    perturbed_beta = beta.at[2].set(0.5)
    perturbed_outputs, _ = tree_verify(
        queries, keys, perturbed_values, decay_factor, perturbed_beta, initial_state, parent_indices
    )

    for node in (0, 1, 3):
        assert_close(result=perturbed_outputs[node], reference=outputs[node])
