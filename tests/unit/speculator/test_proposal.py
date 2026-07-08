import jax
import jax.numpy as jnp
import numpy as np

from lalamo.sampling import SamplingPolicy
from lalamo.speculator import AcceptedProposal, ChainProposal, TreeProposal


def make_proposal() -> ChainProposal:
    return ChainProposal(
        token_ids=jnp.asarray([[10, 11, 12, 13]], dtype=jnp.int32),
        token_positions=jnp.asarray([[5, 6, 7, 8]], dtype=jnp.int32),
        lengths=jnp.asarray([4], dtype=jnp.int32),
    )


def accept(
    proposal: ChainProposal,
    sampled_token_ids: list[int],
    eos_token_ids: list[int] | None = None,
    remaining_lengths: int = 16,
    active: bool = True,
) -> AcceptedProposal:
    result = (
        proposal.accept(jnp.asarray([sampled_token_ids], dtype=jnp.int32))
        .trim_at_eos(jnp.asarray(eos_token_ids or [], dtype=jnp.int32))
        .where_active(jnp.asarray([active]))
    )
    return result.with_lengths(jnp.minimum(result.lengths, jnp.asarray([remaining_lengths], dtype=jnp.int32)))


def test_full_chain_accepted_with_bonus() -> None:
    accepted = accept(make_proposal(), [11, 12, 13, 14])

    assert accepted.token_ids.tolist() == [[11, 12, 13, 14]]
    assert accepted.token_positions.tolist() == [[6, 7, 8, 9]]
    assert accepted.source_indices.tolist() == [[0, 1, 2, 3]]
    assert accepted.lengths.tolist() == [4]
    assert accepted.num_accepted_nodes.tolist() == [4]
    assert accepted.accepted_node_indices.tolist() == [[0, 1, 2, 3]]


def test_partial_chain_accepted() -> None:
    accepted = accept(make_proposal(), [11, 99, 13, 14])

    assert accepted.token_ids.tolist() == [[11, 99, 0, 0]]
    assert accepted.token_positions.tolist() == [[6, 7, 0, 0]]
    assert accepted.source_indices.tolist() == [[0, 1, -1, -1]]
    assert accepted.lengths.tolist() == [2]
    assert accepted.num_accepted_nodes.tolist() == [2]


def test_all_drafts_rejected_emits_bonus_only() -> None:
    accepted = accept(make_proposal(), [99, 98, 97, 96])

    assert accepted.token_ids.tolist() == [[99, 0, 0, 0]]
    assert accepted.token_positions.tolist() == [[6, 0, 0, 0]]
    assert accepted.lengths.tolist() == [1]
    assert accepted.num_accepted_nodes.tolist() == [1]


def test_eos_in_drafts_truncates_and_suppresses_bonus() -> None:
    accepted = accept(make_proposal(), [11, 12, 13, 14], eos_token_ids=[12])

    assert accepted.token_ids.tolist() == [[11, 12, 0, 0]]
    assert accepted.lengths.tolist() == [2]
    assert accepted.has_eos(jnp.asarray([12], dtype=jnp.int32)).tolist() == [True]


def test_eos_as_bonus_token() -> None:
    accepted = accept(make_proposal(), [11, 99, 13, 14], eos_token_ids=[99])

    assert accepted.token_ids.tolist() == [[11, 99, 0, 0]]
    assert accepted.lengths.tolist() == [2]
    assert accepted.has_eos(jnp.asarray([99], dtype=jnp.int32)).tolist() == [True]


def test_remaining_length_caps_emission() -> None:
    accepted = accept(make_proposal(), [11, 12, 13, 14], remaining_lengths=2)

    assert accepted.token_ids.tolist() == [[11, 12, 0, 0]]
    assert accepted.lengths.tolist() == [2]


def test_inactive_line_emits_nothing() -> None:
    accepted = accept(make_proposal(), [11, 12, 13, 14], active=False)

    assert accepted.token_ids.tolist() == [[0, 0, 0, 0]]
    assert accepted.lengths.tolist() == [0]
    assert accepted.num_accepted_nodes.tolist() == [0]


def test_root_only_proposal_emits_bonus() -> None:
    proposal = ChainProposal(
        token_ids=jnp.asarray([[10]], dtype=jnp.int32),
        token_positions=jnp.asarray([[5]], dtype=jnp.int32),
        lengths=jnp.asarray([1], dtype=jnp.int32),
    )
    accepted = accept(proposal, [42])

    assert accepted.token_ids.tolist() == [[42]]
    assert accepted.token_positions.tolist() == [[6]]
    assert accepted.lengths.tolist() == [1]
    assert accepted.num_accepted_nodes.tolist() == [1]


def test_last_token_ids_and_indices() -> None:
    accepted = accept(make_proposal(), [11, 99, 13, 14])
    stopped_token_ids = jnp.asarray([7], dtype=jnp.int32)

    assert accepted.last_token_ids(stopped_token_ids).tolist() == [99]
    assert accepted.last_token_indices().tolist() == [6]

    stopped = accept(make_proposal(), [11, 12, 13, 14], active=False)
    assert stopped.last_token_ids(stopped_token_ids).tolist() == [7]


def test_gather_top_k_follows_source_indices() -> None:
    accepted = accept(make_proposal(), [11, 99, 13, 14])
    node_top_k_token_ids = jnp.asarray([[[1, 2], [3, 4], [5, 6], [7, 8]]], dtype=jnp.int32)
    node_top_k_token_logits = node_top_k_token_ids.astype(jnp.float32)

    token_ids, token_logits = accepted.gather_top_k(node_top_k_token_ids, node_top_k_token_logits)

    assert token_ids.tolist() == [[[1, 2], [3, 4], [0, 0], [0, 0]]]
    assert token_logits.tolist() == [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]]


def build_tree_proposal(
    token_ids: jnp.ndarray,
    token_positions: jnp.ndarray,
    parent_indices: jnp.ndarray,
    lengths: jnp.ndarray,
) -> TreeProposal:
    return TreeProposal(
        token_ids=token_ids,
        token_positions=token_positions,
        parent_indices=parent_indices,
        draft_logprobs=jnp.zeros_like(token_ids, dtype=jnp.float32),
        lengths=lengths,
        max_depth=token_ids.shape[1] - 1,
    )


def make_tree_proposal() -> TreeProposal:
    return build_tree_proposal(
        token_ids=jnp.asarray([[10, 20, 30, 40, 50, 60]], dtype=jnp.int32),
        token_positions=jnp.asarray([[5, 6, 6, 7, 7, 7]], dtype=jnp.int32),
        parent_indices=jnp.asarray([[-1, 0, 0, 1, 2, 2]], dtype=jnp.int32),
        lengths=jnp.asarray([6], dtype=jnp.int32),
    )


def accept_tree(
    proposal: TreeProposal,
    sampled_token_ids: list[int],
    active: bool = True,
) -> AcceptedProposal:
    return proposal.accept(jnp.asarray([sampled_token_ids], dtype=jnp.int32)).where_active(jnp.asarray([active]))


def test_tree_accepts_branch_path_with_bonus() -> None:
    accepted = accept_tree(make_tree_proposal(), [30, 99, 60, 99, 99, 70])

    assert accepted.accepted_node_indices.tolist() == [[0, 2, 5, -1, -1, -1]]
    assert accepted.num_accepted_nodes.tolist() == [3]
    assert accepted.token_ids.tolist() == [[30, 60, 70, 0, 0, 0]]
    assert accepted.token_positions.tolist() == [[6, 7, 8, 0, 0, 0]]
    assert accepted.source_indices.tolist() == [[0, 2, 5, -1, -1, -1]]
    assert accepted.lengths.tolist() == [3]


def test_tree_accepts_other_branch() -> None:
    accepted = accept_tree(make_tree_proposal(), [20, 40, 99, 88, 99, 99])

    assert accepted.accepted_node_indices.tolist() == [[0, 1, 3, -1, -1, -1]]
    assert accepted.token_ids.tolist() == [[20, 40, 88, 0, 0, 0]]
    assert accepted.token_positions.tolist() == [[6, 7, 8, 0, 0, 0]]


def test_tree_rejects_all_children_emits_bonus_only() -> None:
    accepted = accept_tree(make_tree_proposal(), [99, 98, 97, 96, 95, 94])

    assert accepted.accepted_node_indices.tolist() == [[0, -1, -1, -1, -1, -1]]
    assert accepted.token_ids.tolist() == [[99, 0, 0, 0, 0, 0]]
    assert accepted.lengths.tolist() == [1]


def test_tree_duplicate_siblings_resolve_to_lowest_index() -> None:
    proposal = build_tree_proposal(
        token_ids=jnp.asarray([[10, 20, 20]], dtype=jnp.int32),
        token_positions=jnp.asarray([[5, 6, 6]], dtype=jnp.int32),
        parent_indices=jnp.asarray([[-1, 0, 0]], dtype=jnp.int32),
        lengths=jnp.asarray([3], dtype=jnp.int32),
    )
    accepted = accept_tree(proposal, [20, 99, 98])

    assert accepted.accepted_node_indices.tolist() == [[0, 1, -1]]


def test_tree_padded_nodes_are_never_accepted() -> None:
    proposal = build_tree_proposal(
        token_ids=jnp.asarray([[10, 20, 30, 40, 50, 60]], dtype=jnp.int32),
        token_positions=jnp.asarray([[5, 6, 6, 7, 7, 7]], dtype=jnp.int32),
        parent_indices=jnp.asarray([[-1, 0, 0, 1, 2, 2]], dtype=jnp.int32),
        lengths=jnp.asarray([2], dtype=jnp.int32),
    )
    accepted = accept_tree(proposal, [30, 99, 60, 99, 99, 70])

    assert accepted.accepted_node_indices.tolist() == [[0, -1, -1, -1, -1, -1]]
    assert accepted.lengths.tolist() == [1]


def test_tree_inactive_line_emits_nothing() -> None:
    accepted = accept_tree(make_tree_proposal(), [30, 99, 60, 99, 99, 70], active=False)

    assert accepted.token_ids.tolist() == [[0, 0, 0, 0, 0, 0]]
    assert accepted.lengths.tolist() == [0]
    assert accepted.num_accepted_nodes.tolist() == [0]
    assert accepted.accepted_node_indices.tolist() == [[-1, -1, -1, -1, -1, -1]]


def test_tree_with_chain_parents_matches_chain_proposal() -> None:
    chain = make_proposal()
    tree = build_tree_proposal(
        token_ids=chain.token_ids,
        token_positions=chain.token_positions,
        parent_indices=jnp.asarray([[-1, 0, 1, 2]], dtype=jnp.int32),
        lengths=chain.lengths,
    )
    sampled = [11, 12, 99, 98]

    chain_accepted = accept(chain, sampled)
    tree_accepted = accept_tree(tree, sampled)

    assert tree_accepted.token_ids.tolist() == chain_accepted.token_ids.tolist()
    assert tree_accepted.token_positions.tolist() == chain_accepted.token_positions.tolist()
    assert tree_accepted.source_indices.tolist() == chain_accepted.source_indices.tolist()
    assert tree_accepted.lengths.tolist() == chain_accepted.lengths.tolist()
    assert tree_accepted.accepted_node_indices.tolist() == chain_accepted.accepted_node_indices.tolist()
    assert tree_accepted.num_accepted_nodes.tolist() == chain_accepted.num_accepted_nodes.tolist()


def test_tree_verify_greedy_matches_accept_on_random_trees() -> None:
    vocab = 16
    parents = jnp.asarray([[-1, 0, 0, 1, 2, 2, 4]], dtype=jnp.int32)
    depths = jnp.asarray([[0, 1, 1, 2, 2, 2, 3]], dtype=jnp.int32)
    policy = SamplingPolicy.init(temperature=0.0)
    batched_policy = jax.tree.map(lambda leaf: leaf[None], policy)
    key = jax.random.key(0)
    for trial in range(50):
        key, token_key, logit_key = jax.random.split(key, 3)
        token_ids = jax.random.randint(token_key, (1, 7), 0, vocab, dtype=jnp.int32)
        logits = jax.random.normal(logit_key, (1, 7, vocab), dtype=jnp.float32) * 3
        proposal = TreeProposal(
            token_ids=token_ids,
            token_positions=10 + depths,
            parent_indices=parents,
            draft_logprobs=jnp.zeros((1, 7), dtype=jnp.float32),
            lengths=jnp.asarray([7], dtype=jnp.int32),
            max_depth=3,
        )
        processed = jax.vmap(jax.vmap(policy.process_logits))(logits)
        reference = proposal.accept(jnp.argmax(processed, axis=-1).astype(jnp.int32))
        result = proposal.verify(
            processed,
            batched_policy,
            jax.random.split(jax.random.key(trial + 999), (1, 7)),
            jnp.asarray([True]),
            jnp.asarray([0], dtype=jnp.int32),
        )
        assert reference.token_ids.tolist() == result.token_ids.tolist()
        assert reference.lengths.tolist() == result.lengths.tolist()
        assert reference.num_accepted_nodes.tolist() == result.num_accepted_nodes.tolist()


def test_tree_verify_matches_traversal_reference_semantics() -> None:
    vocab = 4
    num_rows = 8000
    token_ids = jnp.broadcast_to(jnp.asarray([[9, 0, 1]], dtype=jnp.int32), (num_rows, 3))
    parents = jnp.broadcast_to(jnp.asarray([[-1, 0, 0]], dtype=jnp.int32), (num_rows, 3))
    positions = jnp.broadcast_to(jnp.asarray([[5, 6, 6]], dtype=jnp.int32), (num_rows, 3))
    draft_logprobs = jnp.broadcast_to(jnp.log(jnp.asarray([[1.0, 0.7, 0.3]], dtype=jnp.float32)), (num_rows, 3))
    root_logits = jnp.log(jnp.asarray([0.45, 0.35, 0.15, 0.05], dtype=jnp.float32))
    logits = jnp.zeros((num_rows, 3, vocab), dtype=jnp.float32).at[:, 0, :].set(root_logits)
    proposal = TreeProposal(
        token_ids=token_ids,
        token_positions=positions,
        parent_indices=parents,
        draft_logprobs=draft_logprobs,
        lengths=jnp.full((num_rows,), 3, dtype=jnp.int32),
        max_depth=1,
    )
    policy = SamplingPolicy.init(temperature=1.0)
    batched_policy = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (num_rows, *leaf.shape)), policy)
    accepted = proposal.verify(
        logits,
        batched_policy,
        jax.random.split(jax.random.key(7), (num_rows, 3)),
        jnp.ones(num_rows, dtype=jnp.bool),
        jnp.zeros(num_rows, dtype=jnp.int32),
    )
    first_tokens = np.asarray(accepted.token_ids[:, 0])
    counts = np.bincount(first_tokens, minlength=vocab)
    target = np.asarray(jax.nn.softmax(root_logits), dtype=np.float64)
    accept_first = min(1.0, target[0] / 0.7)
    accept_second = min(1.0, target[1] / 1.0)
    reject_both = (1 - accept_first) * (1 - accept_second)
    expected = np.asarray(
        [
            accept_first + reject_both * target[0],
            (1 - accept_first) * accept_second + reject_both * target[1],
            reject_both * target[2],
            reject_both * target[3],
        ],
    )
    standard_errors = np.sqrt(expected * (1 - expected) / num_rows)
    assert bool(np.all(np.abs(counts / num_rows - expected) < 5 * standard_errors + 2e-3))
