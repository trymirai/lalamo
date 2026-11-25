"""Tests for the draft tree (`lalamo/modules/speculators/frontier_tree.py`).

The expectations come from the construction as the module docstring states it, recomputed in float64 with plain
loops: which candidates become children, which entries a round expands, which nodes pruning keeps, and how far a
sampled continuation follows the kept nodes. Selections are checked as inequalities with a margin, so a difference in
the last bit of a float32 sum cannot decide a test.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.modules.speculators.frontier_tree import (
    FrontierTree,
    FrontierTreeShape,
    KeptNodes,
    accepted_length,
    build_frontier_tree,
    kept_nodes,
)

SHAPE = FrontierTreeShape(slots=13, children=3, expand=3, budget=8, prune_sigma=1.5)
ROWS, POOL, VOCAB = 4, 12, 500
MARGIN = 1e-4  # float32 path sums of magnitude ~10 carry errors of ~1e-6


def pools(seed: int, batch: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, (batch, ROWS, POOL)).astype(np.float32)
    noise = rng.gumbel(size=(batch, ROWS, POOL)).astype(np.float32)
    tokens = np.stack([np.stack([rng.permutation(VOCAB)[:POOL] for _ in range(ROWS)]) for _ in range(batch)]).astype(
        np.int32
    )
    return logits, noise, tokens, rng.integers(0, VOCAB, batch).astype(np.int32)


def token_correction(carry: None, new, tokens, depth, ancestors):  # noqa: ANN001, ANN201, ARG001
    # Depends on the node only through its token and depth: a pool-shaped wave whose phase they set.
    phase = (tokens * 13 + depth * 7)[..., None].astype(jnp.float32) * 0.1
    return jnp.sin(phase + jnp.arange(POOL, dtype=jnp.float32)), carry


def no_correction(carry: None, new, tokens, depth, ancestors):  # noqa: ANN001, ANN201, ARG001
    return jnp.zeros((*new.shape, POOL), jnp.float32), carry


def build(seed: int, shape: FrontierTreeShape = SHAPE, correction=token_correction, batch: int = 2):  # noqa: ANN001, ANN201
    logits, noise, tokens, root = pools(seed, batch)
    tree = build_frontier_tree(
        shape, jnp.asarray(logits), jnp.asarray(noise), jnp.asarray(tokens), jnp.asarray(root), correction, None
    )
    return tree, logits, noise, tokens, root


def log_sum_exp(values: np.ndarray) -> float:
    top = float(values.max())
    return top + math.log(math.fsum(math.exp(float(value) - top) for value in values))


def path_scores(tree: FrontierTree, logits: np.ndarray, noise: np.ndarray, b: int, sigma: float | None) -> list[float]:
    """Float64 path sums of every node (slots, then entries): sum of min(score_c - logsumexp(score), 0) along the
    path, on l for the expansion weight (sigma None) or on l + G / sigma for pruning."""
    slots, children = tree.slot_token.shape[1], tree.entry_token.shape[1] // tree.slot_token.shape[1]
    slot_depth, slot_entry = np.asarray(tree.slot_depth[b]), np.asarray(tree.slot_entry[b])
    correction, entry_pool = np.asarray(tree.slot_correction[b], np.float64), np.asarray(tree.entry_pool[b])
    slot_score = [0.0] * slots
    entry_score = [-math.inf] * (slots * children)
    for slot in range(slots):
        if slot and slot_entry[slot] >= 0:
            slot_score[slot] = entry_score[slot_entry[slot]]
        depth = int(slot_depth[slot])
        if depth >= ROWS:
            continue
        values = logits[b, depth].astype(np.float64) + correction[slot]
        if sigma is not None:
            values = values + noise[b, depth].astype(np.float64) / sigma
        total = log_sum_exp(values)
        for child in range(children):
            entry = slot * children + child
            entry_score[entry] = slot_score[slot] + min(values[entry_pool[entry]] - total, 0.0)
    return slot_score + entry_score


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_children_of_a_slot_are_the_best_pool_candidates_by_logit_plus_noise(seed: int) -> None:
    # Spec "children": the `children` largest of l + G in the pool of the slot's depth, l = a + rho; entries carry
    # the pool index and the pool's token. Slots at the last depth have no live children.
    tree, logits, noise, tokens, _ = build(seed)
    for b in range(2):
        for slot in range(SHAPE.slots):
            depth = int(tree.slot_depth[b, slot])
            entries = range(slot * SHAPE.children, (slot + 1) * SHAPE.children)
            if depth >= ROWS:
                assert not any(bool(tree.entry_active[b, entry]) for entry in entries)
                continue
            keyed = logits[b, depth].astype(np.float64) + np.asarray(tree.slot_correction[b, slot], np.float64)
            keyed = keyed + noise[b, depth]
            chosen = [int(tree.entry_pool[b, entry]) for entry in entries]
            assert len(set(chosen)) == SHAPE.children
            assert min(keyed[chosen]) >= np.delete(keyed, chosen).max() - MARGIN
            assert list(keyed[chosen]) == sorted(keyed[chosen], reverse=True)
            assert [int(tree.entry_token[b, entry]) for entry in entries] == [int(tokens[b, depth, i]) for i in chosen]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_round_expands_the_heaviest_expandable_frontier_entries(seed: int) -> None:
    # Spec "rounds": slots 1 + (r - 1) * expand .. of round r are the entries with the largest path weight among
    # those written before the round and not yet expanded, entries that may still be expanded (depth < rows) first.
    tree, logits, noise, _, root = build(seed)
    for b in range(2):
        weights = path_scores(tree, logits, noise, b, sigma=None)
        slot_entry, slot_depth = np.asarray(tree.slot_entry[b]), np.asarray(tree.slot_depth[b])
        assert slot_entry[0] == -1 and slot_depth[0] == 0 and int(tree.slot_token[b, 0]) == int(root[b])
        expanded: set[int] = set()
        for round_index in range(1, SHAPE.rounds):
            first = 1 + (round_index - 1) * SHAPE.expand
            written = first * SHAPE.children  # the children of the slots of the earlier rounds
            picked = [int(entry) for entry in slot_entry[first : first + SHAPE.expand]]
            assert all(0 <= entry < written and entry not in expanded for entry in picked)

            def rank(entry: int) -> tuple[bool, float]:
                parent_depth = int(slot_depth[entry // SHAPE.children])  # noqa: B023
                return (parent_depth + 1 < ROWS, weights[SHAPE.slots + entry])  # noqa: B023

            rest = [entry for entry in range(written) if entry not in expanded and entry not in picked]
            if rest:  # round 1 takes all of the root's children when expand equals children
                worst_picked, best_rest = min(rank(entry) for entry in picked), max(rank(entry) for entry in rest)
                assert worst_picked[0] >= best_rest[0]
                if worst_picked[0] == best_rest[0]:
                    assert worst_picked[1] >= best_rest[1] - MARGIN
            expanded.update(picked)
            for offset, entry in enumerate(picked):
                slot = first + offset
                assert slot_depth[slot] == slot_depth[entry // SHAPE.children] + 1
                assert int(tree.slot_token[b, slot]) == int(tree.entry_token[b, entry])
                assert not bool(tree.entry_active[b, entry])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pruning_keeps_the_budget_of_nodes_with_the_largest_noisy_path_score(seed: int) -> None:
    # Spec "pruning": nodes are the slots and the live entries; the kept ones have the largest sum of
    # min(score_c - logsumexp(score), 0) on l + G / sigma; the root is first, parents precede children.
    tree, logits, noise, _, root = build(seed)
    nodes = kept_nodes(SHAPE, tree)
    for b in range(2):
        scores = path_scores(tree, logits, noise, b, sigma=SHAPE.prune_sigma)
        live = [i for i in range(len(scores)) if i < SHAPE.slots or bool(tree.entry_active[b, i - SHAPE.slots])]
        layout_tokens = np.concatenate([np.asarray(tree.slot_token[b]), np.asarray(tree.entry_token[b])])
        best = sorted(live, key=lambda i: -scores[i])
        threshold = scores[best[SHAPE.budget - 1]]
        kept_scores = []
        for position in range(SHAPE.budget):
            assert bool(nodes.valid[b, position])
            candidates = [
                i
                for i in live
                if layout_tokens[i] == int(nodes.token[b, position]) and scores[i] >= threshold - MARGIN
            ]
            assert candidates, "a kept node must be one of the highest-scoring nodes"
            kept_scores.append(max(scores[i] for i in candidates))
        assert min(kept_scores) >= scores[best[SHAPE.budget]] - MARGIN
        assert int(nodes.token[b, 0]) == int(root[b]) and int(nodes.parent[b, 0]) == -1 and int(nodes.depth[b, 0]) == 0
        for position in range(1, SHAPE.budget):
            parent = int(nodes.parent[b, position])
            assert 0 <= parent < position
            assert int(nodes.depth[b, position]) == int(nodes.depth[b, parent]) + 1


def test_a_two_level_tree_computed_by_hand() -> None:
    # One slot per round, two children, no noise, no correction. Pool row 0 (depth 1): logits log(8, 4, 2, 2) ->
    # probabilities 1/2, 1/4, 1/8, 1/8, so the root's children are tokens 10 and 11 with path weights log 1/2 and
    # log 1/4. Round 1 expands token 10; its children come from row 1, logits log(1, 6, 1) -> 1/8, 3/4, 1/8:
    # token 21, then token 20 (the tie of the two 1/8 goes to the smaller pool index), weights log 3/8 and log 1/16.
    # Round 2: token 21 is the heaviest entry left, but it lies at the last depth and entries that may still be
    # expanded go first, so token 11 (log 1/4) becomes the slot; its children are 21 and 20 again, weights log 3/16
    # and log 1/32. With a budget of 4 the kept nodes are the root (0), token 10 (log 1/2), token 21 under 10
    # (log 3/8) and token 11 (log 1/4), listed slots first.
    shape = FrontierTreeShape(slots=3, children=2, expand=1, budget=4, prune_sigma=1.0)
    logits = np.full((1, 2, 4), -1e4, np.float32)
    logits[0, 0] = np.log([8.0, 4.0, 2.0, 2.0])
    logits[0, 1, :3] = np.log([1.0, 6.0, 1.0])
    tokens = np.array([[[10, 11, 12, 13], [20, 21, 22, 23]]], np.int32)

    def none(carry: None, new, tokens, depth, ancestors):  # noqa: ANN001, ANN202, ARG001
        return jnp.zeros((*new.shape, 4), jnp.float32), carry

    tree = build_frontier_tree(
        shape,
        jnp.asarray(logits),
        jnp.zeros((1, 2, 4), jnp.float32),
        jnp.asarray(tokens),
        jnp.asarray([7]),
        none,
        None,
    )
    assert np.asarray(tree.slot_token).tolist() == [[7, 10, 11]]
    assert np.asarray(tree.slot_depth).tolist() == [[0, 1, 1]]
    assert np.asarray(tree.slot_entry).tolist() == [[-1, 0, 1]]
    assert np.asarray(tree.entry_token).tolist() == [[10, 11, 21, 20, 21, 20]]
    assert np.asarray(tree.entry_active).tolist() == [[False, False, True, True, True, True]]
    np.testing.assert_allclose(
        np.asarray(tree.slot_prune)[0], [0.0, math.log(1 / 2), math.log(1 / 4)], atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(tree.entry_prune)[0, 2:],
        [math.log(3 / 8), math.log(1 / 16), math.log(3 / 16), math.log(1 / 32)],
        atol=1e-5,
        rtol=0,
    )
    nodes = kept_nodes(shape, tree)
    assert np.asarray(nodes.token).tolist() == [[7, 10, 11, 21]]
    assert np.asarray(nodes.parent).tolist() == [[-1, 0, 0, 1]]
    assert np.asarray(nodes.depth).tolist() == [[0, 1, 1, 2]]
    assert np.asarray(nodes.valid).all()
    # `sampled` holds the token the target samples after each kept node (root, 10, 11, 21 under 10). 10 after the
    # root and 21 after 10: two drafted tokens accepted. 11 after the root, then a token 11 has no kept child
    # for: one. 20 after the root: none.
    assert accepted_length(nodes, jnp.asarray([[10, 21, 99, 99]])).tolist() == [2]
    assert accepted_length(nodes, jnp.asarray([[11, 21, 99, 99]])).tolist() == [1]
    assert accepted_length(nodes, jnp.asarray([[20, 21, 99, 99]])).tolist() == [0]


def test_a_batch_element_does_not_depend_on_the_rest_of_the_batch() -> None:
    # MR: the construction is per sequence, so a tree built alone equals its row of a batch.
    tree, logits, noise, tokens, root = build(3, batch=3)
    alone = build_frontier_tree(
        SHAPE,
        jnp.asarray(logits[1:2]),
        jnp.asarray(noise[1:2]),
        jnp.asarray(tokens[1:2]),
        jnp.asarray(root[1:2]),
        token_correction,
        None,
    )
    for field, value in zip(tree._fields, tree, strict=True):
        np.testing.assert_array_equal(np.asarray(value[1:2]), np.asarray(getattr(alone, field)), err_msg=field)


def test_the_kept_tokens_follow_a_reordering_of_the_pool() -> None:
    # MR: the tree is a function of the candidates, not of their order in the pool (no ties in random floats).
    tree, logits, noise, tokens, root = build(4, correction=no_correction)
    order = np.random.default_rng(5).permutation(POOL)
    shuffled = build_frontier_tree(
        SHAPE,
        jnp.asarray(logits[..., order]),
        jnp.asarray(noise[..., order]),
        jnp.asarray(tokens[..., order]),
        jnp.asarray(root),
        no_correction,
        None,
    )
    np.testing.assert_array_equal(np.asarray(tree.slot_token), np.asarray(shuffled.slot_token))
    np.testing.assert_array_equal(np.asarray(tree.entry_token), np.asarray(shuffled.entry_token))
    np.testing.assert_array_equal(
        np.asarray(kept_nodes(SHAPE, tree).token), np.asarray(kept_nodes(SHAPE, shuffled).token)
    )


def test_without_noise_and_correction_the_first_slot_is_the_most_likely_token() -> None:
    # Limit: with G = 0 and rho = 0 the heaviest child of the root is the arg max of the first pool row.
    logits, _, tokens, root = pools(6)
    tree = build_frontier_tree(
        SHAPE,
        jnp.asarray(logits),
        jnp.zeros(logits.shape, jnp.float32),
        jnp.asarray(tokens),
        jnp.asarray(root),
        no_correction,
        None,
    )
    expected = np.take_along_axis(tokens[:, 0], logits[:, 0].argmax(axis=-1)[:, None], axis=1)[:, 0]
    np.testing.assert_array_equal(np.asarray(tree.slot_token[:, 1]), expected)


def test_acceptance_follows_the_sampled_tokens_through_valid_children_only() -> None:
    # Spec: the length of the path on which each node's child carries the token sampled after the node.
    nodes = KeptNodes(
        token=jnp.asarray([[5, 6, 7, 8, 9, 6]]),
        depth=jnp.asarray([[0, 1, 1, 2, 3, 2]]),
        parent=jnp.asarray([[-1, 0, 0, 1, 3, 2]]),
        valid=jnp.asarray([[True, True, True, True, True, False]]),
    )
    assert accepted_length(nodes, jnp.asarray([[6, 8, 0, 9, 1, 0]])).tolist() == [3]  # 5 -> 6 -> 8 -> 9
    assert accepted_length(nodes, jnp.asarray([[6, 8, 0, 4, 1, 0]])).tolist() == [2]  # 9 is not sampled after 8
    assert accepted_length(nodes, jnp.asarray([[7, 8, 6, 9, 1, 0]])).tolist() == [1]  # node 5 is invalid
    assert accepted_length(nodes, jnp.asarray([[4, 8, 0, 9, 1, 0]])).tolist() == [0]
    no_root = nodes._replace(valid=jnp.zeros((1, 6), bool))
    assert accepted_length(no_root, jnp.asarray([[6, 8, 0, 9, 1, 0]])).tolist() == [0]


def test_a_budget_above_the_node_count_marks_the_missing_nodes_invalid() -> None:
    # Spec: `valid` is False where the tree holds fewer nodes than the budget. One row of pools: the root and its
    # three children are the only nodes with a finite score, whatever the number of slots.
    shape = FrontierTreeShape(slots=4, children=3, expand=3, budget=9, prune_sigma=1.5)
    logits, noise, tokens, root = pools(7)
    tree = build_frontier_tree(
        shape,
        jnp.asarray(logits[:, :1]),
        jnp.asarray(noise[:, :1]),
        jnp.asarray(tokens[:, :1]),
        jnp.asarray(root),
        token_correction,
        None,
    )
    nodes = kept_nodes(shape, tree)
    assert np.asarray(nodes.valid).sum(axis=1).tolist() == [4, 4]
    assert (np.asarray(nodes.parent)[~np.asarray(nodes.valid)] == -1).all()


@pytest.mark.parametrize(
    "fields",
    [
        {"slots": 0},
        {"slots": 6, "expand": 4},
        {"budget": 0},
        {"slots": 5, "children": 2, "expand": 4, "budget": 16},
        {"prune_sigma": 0.0},
    ],
)
def test_impossible_shapes_are_rejected(fields: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FrontierTreeShape(**fields)  # type: ignore[arg-type]
