import jax.numpy as jnp

from lalamo.modules.token_mixers.state import State, StaticKVCacheLayer
from lalamo.modules.token_mixers.state.kv_cache import compact_state_layers


def test_static_kv_cache_compact_moves_accepted_draft_slots() -> None:
    keys = jnp.arange(6, dtype=jnp.float32).reshape(1, 6, 1, 1)
    values = keys + 100
    layer = StaticKVCacheLayer(
        has_sinks=False,
        keys=keys,
        values=values,
        current_length=jnp.array([6], dtype=jnp.int32),
    )

    compacted_state = compact_state_layers(
        State([layer]),
        cache_len=jnp.array(2, dtype=jnp.int32),
        accepted_indices=jnp.array([2, 0, 3, 1], dtype=jnp.int32),
        num_accepted=jnp.array(2, dtype=jnp.int32),
        max_slots=4,
    )

    compacted_layer = compacted_state[0]
    assert isinstance(compacted_layer, StaticKVCacheLayer)
    assert jnp.array_equal(
        compacted_layer.keys[:, :, 0, 0],
        jnp.array([[0, 1, 4, 2, 4, 5]], dtype=jnp.float32),
    )
    assert jnp.array_equal(
        compacted_layer.values[:, :, 0, 0],
        jnp.array([[100, 101, 104, 102, 104, 105]], dtype=jnp.float32),
    )
    assert jnp.array_equal(compacted_layer.current_length, jnp.array([4], dtype=jnp.int32))
    assert compacted_layer.prefix_length_for_sample(0) == 4
