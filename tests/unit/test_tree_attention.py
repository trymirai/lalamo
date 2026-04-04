import jax.numpy as jnp
import numpy as np

from lalamo.speculator.tree_attention import build_tree_mask


class TestBuildTreeMask:
    def test_single_root_chain(self) -> None:
        """Linear chain: root -> child -> grandchild."""
        # draft[0]: root (parent=-1)
        # draft[1]: child of draft[0]
        # draft[2]: child of draft[1]
        parent_indices = jnp.array([-1, 0, 1], dtype=jnp.int32)
        prefix_length = 4
        capacity = 10

        mask = build_tree_mask(parent_indices, prefix_length, capacity)
        assert mask.shape == (3, 10)

        # All draft tokens attend to prefix (columns 0-3)
        np.testing.assert_array_equal(mask[:, :4], True)

        # Unused capacity (columns 7-9) not attended
        np.testing.assert_array_equal(mask[:, 7:], False)

        # draft[0] (col 4): attends to self only among draft
        assert mask[0, 4] == True
        assert mask[0, 5] == False
        assert mask[0, 6] == False

        # draft[1] (col 5): attends to draft[0] and self
        assert mask[1, 4] == True
        assert mask[1, 5] == True
        assert mask[1, 6] == False

        # draft[2] (col 6): attends to draft[0], draft[1], and self
        assert mask[2, 4] == True
        assert mask[2, 5] == True
        assert mask[2, 6] == True

    def test_binary_tree(self) -> None:
        """Binary tree: two roots, each with one child."""
        # draft[0]: root A
        # draft[1]: root B
        # draft[2]: child of A
        # draft[3]: child of B
        parent_indices = jnp.array([-1, -1, 0, 1], dtype=jnp.int32)
        prefix_length = 2
        capacity = 8

        mask = build_tree_mask(parent_indices, prefix_length, capacity)

        # Prefix attended by all
        np.testing.assert_array_equal(mask[:, :2], True)

        # draft[2] (col 4): attends to draft[0] (col 2) and self (col 4), NOT draft[1] (col 3)
        assert mask[2, 2] == True   # parent (draft[0])
        assert mask[2, 3] == False  # sibling root (draft[1])
        assert mask[2, 4] == True   # self

        # draft[3] (col 5): attends to draft[1] (col 3) and self (col 5), NOT draft[0] (col 2)
        assert mask[3, 2] == False  # other root (draft[0])
        assert mask[3, 3] == True   # parent (draft[1])
        assert mask[3, 5] == True   # self

    def test_all_roots(self) -> None:
        """All draft tokens are roots (no tree structure)."""
        parent_indices = jnp.array([-1, -1, -1], dtype=jnp.int32)
        prefix_length = 3
        capacity = 8

        mask = build_tree_mask(parent_indices, prefix_length, capacity)

        # Each root attends to prefix + self only
        for i in range(3):
            np.testing.assert_array_equal(mask[i, :3], True)  # prefix
            assert mask[i, 3 + i] == True  # self
            for j in range(3):
                if j != i:
                    assert mask[i, 3 + j] == False  # other roots

    def test_prefix_length_zero(self) -> None:
        """Edge case: no prefix tokens."""
        parent_indices = jnp.array([-1, 0], dtype=jnp.int32)
        prefix_length = 0
        capacity = 4

        mask = build_tree_mask(parent_indices, prefix_length, capacity)

        # draft[0]: self only
        assert mask[0, 0] == True
        assert mask[0, 1] == False

        # draft[1]: parent (draft[0]) + self
        assert mask[1, 0] == True
        assert mask[1, 1] == True

    def test_unused_capacity_is_false(self) -> None:
        parent_indices = jnp.array([-1], dtype=jnp.int32)
        prefix_length = 2
        capacity = 16

        mask = build_tree_mask(parent_indices, prefix_length, capacity)

        # Only prefix (0,1) and self (2) should be True
        assert mask[0, 2] == True
        np.testing.assert_array_equal(mask[0, 3:], False)
