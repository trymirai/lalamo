import jax.numpy as jnp

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


def make_tree_proposal() -> TreeProposal:
    return TreeProposal(
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
    proposal = TreeProposal(
        token_ids=jnp.asarray([[10, 20, 20]], dtype=jnp.int32),
        token_positions=jnp.asarray([[5, 6, 6]], dtype=jnp.int32),
        parent_indices=jnp.asarray([[-1, 0, 0]], dtype=jnp.int32),
        lengths=jnp.asarray([3], dtype=jnp.int32),
    )
    accepted = accept_tree(proposal, [20, 99, 98])

    assert accepted.accepted_node_indices.tolist() == [[0, 1, -1]]


def test_tree_padded_nodes_are_never_accepted() -> None:
    proposal = TreeProposal(
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
    tree = TreeProposal(
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
