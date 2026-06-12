import jax.numpy as jnp

from lalamo.speculator import AcceptedProposal, ChainProposal


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
    return proposal.accept(
        sampled_token_ids=jnp.asarray([sampled_token_ids], dtype=jnp.int32),
        remaining_lengths=jnp.asarray([remaining_lengths], dtype=jnp.int32),
        eos_token_ids=jnp.asarray(eos_token_ids or [99999], dtype=jnp.int32),
        active_mask=None if active else jnp.asarray([False]),
    )


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
