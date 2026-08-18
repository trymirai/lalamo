import jax
import jax.numpy as jnp
import pytest

from lalamo.modules.token_mixer import MixerForwardPassConfig
from lalamo.modules.token_mixers.attention import _append_paged_key_values, _composed_paged_attention
from lalamo.modules.token_mixers.kv_cache import StaticKVCacheLayer


@pytest.mark.parametrize("sliding_window_size", [None, 1, 2, 3])
@pytest.mark.parametrize("has_sinks", [False, True])
def test_composed_paged_attention_matches_dense_mask_for_permuted_pages_lengths_and_windows(
    sliding_window_size: int | None,
    has_sinks: bool,
) -> None:
    queries = jnp.array(
        [
            [[1.0, 0.5], [0.25, 1.0]],
            [[0.5, 1.0], [1.0, 0.25]],
        ],
        dtype=jnp.float32,
    )
    key_pages = jnp.array(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [2.0, 0.0]],
                [[0.0, 2.0], [2.0, 2.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    value_pages = key_pages + 0.5
    block_tables = jnp.array([[2, 0], [1, 2]], dtype=jnp.int32)
    lengths = jnp.array([3, 4], dtype=jnp.int32)
    sinks = jnp.array([0.3, -0.2], dtype=jnp.float32) if has_sinks else None
    if has_sinks:
        key_pages = key_pages.at[:, block_tables[:, 0], 0].set(0.0)
        value_pages = value_pages.at[:, block_tables[:, 0], 0].set(0.0)

    actual = _composed_paged_attention(
        queries,
        key_pages,
        value_pages,
        block_tables,
        lengths,
        scale=0.5,
        logit_soft_cap=None,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        forward_pass_config=MixerForwardPassConfig(),
    )

    gathered_keys = jnp.stack(
        [
            jnp.concatenate((key_pages[0, 2], key_pages[0, 0])),
            jnp.concatenate((key_pages[0, 1], key_pages[0, 2])),
        ]
    )
    gathered_values = jnp.stack(
        [
            jnp.concatenate((value_pages[0, 2], value_pages[0, 0])),
            jnp.concatenate((value_pages[0, 1], value_pages[0, 2])),
        ]
    )

    def dense_decode_mask(keys: jax.Array, values: jax.Array, length: jax.Array) -> jax.Array:
        # The dense static cache (already extended with the query token) defines the decode mask contract.
        return StaticKVCacheLayer(
            has_sinks=has_sinks,
            keys=keys[:, None, :],
            values=values[:, None, :],
            current_length=length,
        ).attention_mask(suffix_length=1, is_causal=True, sliding_window_size=sliding_window_size)[0]

    mask = jax.vmap(dense_decode_mask)(gathered_keys, gathered_values, lengths)
    if sinks is None:
        sink_bias = None
    else:
        sink_bias = jnp.zeros((queries.shape[1], 1, gathered_keys.shape[1]), dtype=queries.dtype)
        sink_bias = sink_bias.at[:, :, 0].set(sinks[:, None])
    expected = jax.vmap(
        lambda query, keys, values, row_mask: jax.nn.dot_product_attention(
            query[None, :, :],
            keys[:, None, :],
            values[:, None, :],
            bias=sink_bias,
            mask=row_mask[None, :],
            scale=0.5,
        )[0]
    )(queries, gathered_keys, gathered_values, mask)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_append_paged_key_values_writes_each_sequence_to_its_logical_tail() -> None:
    key_pages = jnp.zeros((2, 5, 4, 3), dtype=jnp.float32)
    value_pages = jnp.zeros_like(key_pages)
    block_tables = jnp.array([[3, 1], [4, 2]], dtype=jnp.int32)
    lengths = jnp.array([2, 5], dtype=jnp.int32)
    added_keys = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
    added_values = added_keys + 100.0

    updated_keys, updated_values = _append_paged_key_values(
        key_pages,
        value_pages,
        block_tables,
        lengths,
        added_keys,
        added_values,
    )

    assert jnp.array_equal(updated_keys[:, 3, 2], added_keys[0])
    assert jnp.array_equal(updated_keys[:, 2, 1], added_keys[1])
    assert jnp.array_equal(updated_values[:, 3, 2], added_values[0])
    assert jnp.array_equal(updated_values[:, 2, 1], added_values[1])
    assert jnp.count_nonzero(updated_keys).item() == jnp.count_nonzero(added_keys).item()
