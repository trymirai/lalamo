import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.modules.token_mixers.state.kv_cache import StaticKVCacheLayer


def _make_cache(capacity: int = 16, num_groups: int = 2, head_dim: int = 4) -> StaticKVCacheLayer:
    return StaticKVCacheLayer.init(
        has_sinks=False,
        capacity=capacity,
        num_groups=num_groups,
        head_dim=head_dim,
        dtype=jnp.float32,
    )


def _make_kv(num_tokens: int, num_groups: int = 2, head_dim: int = 4, *, offset: float = 0.0) -> tuple:
    keys = jnp.broadcast_to(
        jnp.arange(num_tokens, dtype=jnp.float32)[:, None, None] + offset,
        (num_tokens, num_groups, head_dim),
    )
    values = keys + 100.0
    return keys, values


class TestExtendUnchanged:
    def test_sequential_extend_preserves_current_length(self) -> None:
        cache = _make_cache()
        keys, values = _make_kv(3)
        cache = cache.extend(keys, values)
        assert int(cache.current_length) == 3

        keys2, values2 = _make_kv(2, offset=10.0)
        cache = cache.extend(keys2, values2)
        assert int(cache.current_length) == 5

    def test_causal_mask_shape_unchanged(self) -> None:
        cache = _make_cache(capacity=8)
        keys, values = _make_kv(4)
        cache = cache.extend(keys, values)
        mask = cache.attention_mask(suffix_length=2, is_causal=True)
        assert mask.shape == (2, 8)


class TestTreeMaskPassthrough:
    def test_tree_mask_returned_as_is(self) -> None:
        cache = _make_cache(capacity=8)
        keys, values = _make_kv(4)
        cache = cache.extend(keys, values)

        tree_mask = jnp.eye(2, 8, dtype=jnp.bool)
        result = cache.attention_mask(
            suffix_length=2, is_causal=True, tree_mask=tree_mask,
        )
        np.testing.assert_array_equal(result, tree_mask)

    def test_none_tree_mask_falls_through_to_causal(self) -> None:
        cache = _make_cache(capacity=8)
        keys, values = _make_kv(4)
        cache = cache.extend(keys, values)

        mask_default = cache.attention_mask(suffix_length=2, is_causal=True)
        mask_none = cache.attention_mask(suffix_length=2, is_causal=True, tree_mask=None)
        np.testing.assert_array_equal(mask_default, mask_none)


class TestRepack:
    def test_repack_current_length(self) -> None:
        cache = _make_cache(capacity=16)
        prefix_keys, prefix_values = _make_kv(4)
        cache = cache.extend(prefix_keys, prefix_values)
        prefix_length = cache.current_length

        draft_keys, draft_values = _make_kv(5, offset=10.0)
        cache = cache.extend(draft_keys, draft_values)
        assert int(cache.current_length) == 9

        # Accept 3 out of 5 draft tokens (indices 0, 2, 4)
        accepted_indices = jnp.array([0, 2, 4], dtype=jnp.int32)
        accepted_count = jnp.array(3, dtype=jnp.int32)
        cache = cache.repack(accepted_indices, accepted_count, prefix_length)
        assert int(cache.current_length) == 4 + 3

    def test_repack_kv_entries_correct(self) -> None:
        cache = _make_cache(capacity=16, num_groups=1, head_dim=1)

        # Prefix: values [0, 1, 2]
        prefix_keys = jnp.array([[[0.0]], [[1.0]], [[2.0]]])
        prefix_values = jnp.array([[[100.0]], [[101.0]], [[102.0]]])
        cache = cache.extend(prefix_keys, prefix_values)
        prefix_length = cache.current_length

        # Draft: values [10, 11, 12, 13]
        draft_keys = jnp.array([[[10.0]], [[11.0]], [[12.0]], [[13.0]]])
        draft_values = jnp.array([[[110.0]], [[111.0]], [[112.0]], [[113.0]]])
        cache = cache.extend(draft_keys, draft_values)

        # Accept draft indices 1 and 3 (keys 11.0 and 13.0)
        accepted_indices = jnp.array([1, 3], dtype=jnp.int32)
        accepted_count = jnp.array(2, dtype=jnp.int32)
        cache = cache.repack(accepted_indices, accepted_count, prefix_length)

        # Prefix untouched
        np.testing.assert_allclose(cache.keys[0, 0, 0], 0.0)
        np.testing.assert_allclose(cache.keys[1, 0, 0], 1.0)
        np.testing.assert_allclose(cache.keys[2, 0, 0], 2.0)

        # Accepted draft entries packed at positions 3 and 4
        np.testing.assert_allclose(cache.keys[3, 0, 0], 11.0)
        np.testing.assert_allclose(cache.keys[4, 0, 0], 13.0)

        np.testing.assert_allclose(cache.values[3, 0, 0], 111.0)
        np.testing.assert_allclose(cache.values[4, 0, 0], 113.0)

    def test_repack_accept_none(self) -> None:
        cache = _make_cache(capacity=16)
        prefix_keys, prefix_values = _make_kv(4)
        cache = cache.extend(prefix_keys, prefix_values)
        prefix_length = cache.current_length

        draft_keys, draft_values = _make_kv(3, offset=10.0)
        cache = cache.extend(draft_keys, draft_values)

        # Accept 0 tokens — current_length resets to prefix_length
        accepted_indices = jnp.array([0], dtype=jnp.int32)  # dummy, not used
        accepted_count = jnp.array(0, dtype=jnp.int32)
        cache = cache.repack(accepted_indices, accepted_count, prefix_length)
        assert int(cache.current_length) == 4
