import jax.numpy as jnp
from jaxtyping import Array, Bool, Int


def build_tree_mask(
    parent_indices: Int[Array, " num_draft"],
    prefix_length: Int[Array, ""] | int,
    capacity: int,
) -> Bool[Array, "num_draft capacity"]:
    """Build an attention mask for tree-structured draft tokens.

    Each draft token attends to:
      - All prefix tokens (columns 0..prefix_length-1)
      - Its ancestor chain in the tree (determined by parent_indices)
      - Itself

    Args:
        parent_indices: For each draft token, the index of its parent among draft tokens.
            -1 means the token is a root (parent is the prefix).
        prefix_length: Number of prefix tokens already in the KV cache.
        capacity: Total KV cache capacity.

    Returns:
        Boolean mask of shape (num_draft, capacity).
    """
    num_draft = parent_indices.shape[0]
    prefix_length = jnp.asarray(prefix_length, dtype=jnp.int32)

    # Prefix part: all draft tokens attend to all prefix positions
    col_indices = jnp.arange(capacity, dtype=jnp.int32)
    prefix_mask = col_indices[None, :] < prefix_length  # (1, capacity) broadcast

    # Draft-to-draft part: compute ancestor matrix via repeated parent chasing
    # ancestor_of[i, j] = True iff draft j is an ancestor of draft i (or j == i)
    draft_indices = jnp.arange(num_draft, dtype=jnp.int32)

    # Self-attend: each token is its own ancestor
    ancestor_matrix = jnp.eye(num_draft, dtype=jnp.bool)

    # Chase parent pointers up to num_draft times (max possible depth)
    current = parent_indices  # current[i] = parent of draft i
    for _ in range(num_draft):
        # Mark current[i] as ancestor of i (if current[i] >= 0)
        valid = current >= 0
        ancestor_matrix = ancestor_matrix | (
            valid[:, None] & (draft_indices[None, :] == current[:, None])
        )
        # Move up: current[i] = parent_indices[current[i]] (if valid)
        current = jnp.where(valid, parent_indices[current], -1)

    # Place draft ancestor matrix into the correct columns of the full mask
    # Draft tokens occupy columns prefix_length..prefix_length+num_draft-1
    draft_col_offsets = col_indices - prefix_length  # which draft index each column maps to
    in_draft_range = (draft_col_offsets >= 0) & (draft_col_offsets < num_draft)

    # For each (row, col), look up ancestor_matrix[row, draft_col_offsets[col]]
    # Clamp to valid range for indexing, then mask with in_draft_range
    clamped_offsets = jnp.clip(draft_col_offsets, 0, num_draft - 1)
    draft_mask = ancestor_matrix[:, clamped_offsets] & in_draft_range[None, :]

    return prefix_mask | draft_mask
