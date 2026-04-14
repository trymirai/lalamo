import jax.numpy as jnp
import numpy as np

from lalamo.models.batch_scheduler import append_block_tokens


def test_append_block_tokens_drops_overgenerated_writes() -> None:
    num_lines = 2
    max_output = 6
    block_size = 4

    sentinel = -42
    token_buffer = jnp.full((num_lines, max_output), sentinel, dtype=jnp.int32)
    # line 0 has room for the whole block; line 1 is already at the edge.
    num_generated = jnp.array([0, max_output], dtype=jnp.int32)
    block_tokens = jnp.arange(block_size * num_lines, dtype=jnp.int32).reshape(block_size, num_lines)

    new_buffer, new_num_generated = append_block_tokens(token_buffer, num_generated, block_tokens)

    assert np.array_equal(np.asarray(new_num_generated), np.array([block_size, max_output + block_size]))

    new_buffer_np = np.asarray(new_buffer)
    expected_line0 = np.array([0, 2, 4, 6, sentinel, sentinel], dtype=np.int32)
    assert np.array_equal(new_buffer_np[0], expected_line0)
    expected_line1 = np.full(max_output, sentinel, dtype=np.int32)
    assert np.array_equal(new_buffer_np[1], expected_line1)


def test_append_block_tokens_partial_overflow_preserves_last_slot() -> None:
    max_output = 5
    token_buffer = jnp.zeros((1, max_output), dtype=jnp.int32)
    num_generated = jnp.array([3], dtype=jnp.int32)
    block_tokens = jnp.array([[10], [20], [30], [40]], dtype=jnp.int32)

    new_buffer, _ = append_block_tokens(token_buffer, num_generated, block_tokens)
    new_buffer_np = np.asarray(new_buffer)[0]

    assert new_buffer_np[3] == 10
    assert new_buffer_np[4] == 20
