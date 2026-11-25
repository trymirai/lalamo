"""Tests for `TreeSetRouting`: one set of experts for a draft tree verified as a batch of root-to-node paths.

The set is checked against a plain-Python greedy written from the docstring (sets, loops, float64), and against
`BlockSetRouting`, which has tests of its own, in the layout where the two rules describe the same problem: every
sequence holds one active token, so the nodes of the tree are the tokens of one block.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.modules.mlp import RoutingFunction, RoutingIntervention, SoftmaxRouting
from lalamo.modules.routing_interventions import BlockObjective, BlockSetRouting, BlockState, TreeSetRouting
from tests.unit.modules.test_block_set_routing import own_routing
from tests.unit.modules.test_scored_cache_routing import (
    BOTH_PHASES,
    NUM_ACTIVE,
    NUM_EXPERTS,
    ROUTING_FUNCTIONS,
    _biases,
    _logits,
)

TOKENS, BATCH = 5, 6


def run(
    rule: TreeSetRouting, routing_function: RoutingFunction, logits: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    biases = _biases(routing_function)
    routing, state = rule.route_block(
        jnp.asarray(logits),
        jnp.asarray(active),
        None,
        routing_function,
        NUM_ACTIVE,
        None if biases is None else jnp.asarray(biases),
    )
    assert state is None and routing.effective_logits is not None
    return (
        np.asarray(routing.active_expert_indices),
        np.asarray(routing.active_expert_weights),
        np.asarray(routing.effective_logits),
    )


def paths(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random logits [tokens, batch, experts] and paths: sequence b is active on a run of tokens that ends at a
    random place (so its node is not always the last token of the call); one sequence is empty."""
    rng = np.random.default_rng(seed)
    logits = _logits((TOKENS, BATCH, NUM_EXPERTS), seed)
    active = np.zeros((TOKENS, BATCH), dtype=bool)
    for b in range(BATCH - 1):
        end = int(rng.integers(1, TOKENS + 1))
        start = int(rng.integers(0, end))
        active[start:end, b] = True
    return logits, active


def reference_set(
    rule: TreeSetRouting, routing_function: RoutingFunction, logits: np.ndarray, active: np.ndarray
) -> set[int]:
    """The greedy of the docstring over the last active token of every sequence, in plain Python."""
    nodes = [(int(np.flatnonzero(active[:, b])[-1]), b) for b in range(active.shape[1]) if active[:, b].any()]
    rows = np.stack([logits[t, b] for t, b in nodes])
    indices, _, scores = own_routing(routing_function, rows)
    scores = scores.astype(np.float64)
    values = []
    for row in range(len(nodes)):
        above = scores[row] - scores[row].min()
        values.append(above / sum(above[expert] for expert in indices[row].tolist()))
    forced: set[int] = set()
    for row in range(len(nodes)):
        forced.update(sorted(range(NUM_EXPERTS), key=lambda e: (-scores[row][e], e))[: rule.forced_top])
    chosen: set[int] = set()

    def total(candidate: set[int]) -> float:
        return sum(sum(sorted((row[e] for e in candidate), reverse=True)[:NUM_ACTIVE]) for row in values)

    while len(chosen) < min(rule.max_size, NUM_EXPERTS):
        gains = {e: total(chosen | {e}) - total(chosen) for e in range(NUM_EXPERTS) if e not in chosen}
        floor_left = forced - chosen
        pool = floor_left or set(gains)
        best = min(pool, key=lambda e: (-gains[e], e))
        if not floor_left and len(chosen) >= NUM_ACTIVE and max(gains.values()) < rule.threshold:
            break
        chosen.add(best)
    return chosen


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("threshold,forced_top", [(0.05, 0), (0.2, 0), (0.2, 2), (0.6, 1)])
def test_the_set_is_the_greedy_over_the_nodes_and_binds_every_active_token(
    routing_function: RoutingFunction, threshold: float, forced_top: int
) -> None:
    # Spec: the set comes from the last active token of every sequence; every active token is routed to its best
    # experts inside the set; other tokens keep the model's own routing.
    rule = TreeSetRouting(phases=BOTH_PHASES, threshold=threshold, max_size=NUM_EXPERTS, forced_top=forced_top)
    for seed in range(3):
        logits, active = paths(seed)
        indices, weights, effective = run(rule, routing_function, logits, active)
        expected = reference_set(rule, routing_function, logits, active)
        own_indices, own_weights, scores = own_routing(routing_function, logits.reshape(-1, NUM_EXPERTS))
        own_indices = own_indices.reshape(TOKENS, BATCH, NUM_ACTIVE)
        own_weights = own_weights.reshape(TOKENS, BATCH, NUM_ACTIVE)
        scores = scores.reshape(TOKENS, BATCH, NUM_EXPERTS)
        for t in range(TOKENS):
            for b in range(BATCH):
                if not active[t, b]:
                    assert set(indices[t, b].tolist()) == set(own_indices[t, b].tolist())
                    np.testing.assert_array_equal(weights[t, b], own_weights[t, b])
                    continue
                assert set(np.flatnonzero(np.isfinite(effective[t, b])).tolist()) == expected
                best = sorted(expected, key=lambda e: (-scores[t, b][e], e))[:NUM_ACTIVE]
                assert set(indices[t, b].tolist()) == set(best)


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_single_token_paths_are_one_block_of_the_block_rule(routing_function: RoutingFunction) -> None:
    # Differential: with one active token per sequence the nodes are exactly the tokens, and the block rule over
    # the same tokens laid out as one sequence and one block must choose the same routing.
    rule = TreeSetRouting(phases=BOTH_PHASES, threshold=0.25, max_size=20, forced_top=1)
    block = BlockSetRouting(
        phases=BOTH_PHASES, window=BATCH, threshold=0.25, objective=BlockObjective.SCORE, max_size=20, forced_top=1
    )
    logits = _logits((1, BATCH, NUM_EXPERTS), 11)
    indices, weights, effective = run(rule, routing_function, logits, np.ones((1, BATCH), dtype=bool))
    biases = _biases(routing_function)
    routing, _ = block.route_block(
        jnp.asarray(logits.reshape(BATCH, 1, NUM_EXPERTS)),
        jnp.ones((BATCH, 1), dtype=bool),
        BlockState(active_tokens=jnp.zeros((1,), dtype=jnp.int32)),
        routing_function,
        NUM_ACTIVE,
        None if biases is None else jnp.asarray(biases),
    )
    np.testing.assert_array_equal(indices[0], np.asarray(routing.active_expert_indices)[:, 0])
    np.testing.assert_array_equal(weights[0], np.asarray(routing.active_expert_weights)[:, 0])
    np.testing.assert_array_equal(effective[0], np.asarray(routing.effective_logits)[:, 0])


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_repeated_ancestors_and_the_order_of_the_paths_do_not_change_the_set(
    routing_function: RoutingFunction,
) -> None:
    # MR: a node counts once however many paths repeat it, so what stands before the last active token of a
    # sequence cannot move the set; and the set is a function of the set of nodes, not of their order.
    rule = TreeSetRouting(phases=BOTH_PHASES, threshold=0.15, max_size=24, forced_top=0)
    logits, active = paths(21)
    _, _, effective = run(rule, routing_function, logits, active)
    in_set = np.isfinite(effective[active][0])
    scrambled = logits.copy()
    for b in range(BATCH):
        rows = np.flatnonzero(active[:, b])
        if rows.size > 1:
            scrambled[rows[:-1], b] = _logits((rows.size - 1, NUM_EXPERTS), 100 + b)
    _, _, effective_scrambled = run(rule, routing_function, scrambled, active)
    np.testing.assert_array_equal(np.isfinite(effective_scrambled[active][0]), in_set)
    order = np.random.default_rng(3).permutation(BATCH)
    _, _, effective_permuted = run(rule, routing_function, logits[:, order], active[:, order])
    np.testing.assert_array_equal(np.isfinite(effective_permuted[active[:, order]][0]), in_set)


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_a_threshold_nothing_falls_below_is_the_model_routing_at_the_nodes(routing_function: RoutingFunction) -> None:
    # Neutral element: with threshold 0 the set grows while any node gains, so it holds every expert a node
    # selects and the nodes keep the model's own experts.
    rule = TreeSetRouting(phases=BOTH_PHASES, threshold=0.0, max_size=NUM_EXPERTS)
    logits, active = paths(31)
    indices, _, _ = run(rule, routing_function, logits, active)
    own_indices, _, _ = own_routing(routing_function, logits.reshape(-1, NUM_EXPERTS))
    own_indices = own_indices.reshape(TOKENS, BATCH, NUM_ACTIVE)
    for b in range(BATCH):
        rows = np.flatnonzero(active[:, b])
        if rows.size:
            assert set(indices[rows[-1], b].tolist()) == set(own_indices[rows[-1], b].tolist())


def test_a_call_without_active_tokens_keeps_the_model_routing() -> None:
    routing_function = SoftmaxRouting()
    rule = TreeSetRouting(phases=BOTH_PHASES, threshold=0.3)
    logits = _logits((TOKENS, 2, NUM_EXPERTS), 41)
    indices, weights, _ = run(rule, routing_function, logits, np.zeros((TOKENS, 2), dtype=bool))
    own_indices, own_weights, _ = own_routing(routing_function, logits.reshape(-1, NUM_EXPERTS))
    np.testing.assert_array_equal(indices.reshape(-1, NUM_ACTIVE), own_indices)
    np.testing.assert_array_equal(weights.reshape(-1, NUM_ACTIVE), own_weights)


def test_the_rule_validates_its_parameters_has_no_single_token_step_and_round_trips_through_json() -> None:
    for fields in ({"threshold": -0.1}, {"threshold": 0.1, "max_size": 0}, {"threshold": 0.1, "forced_top": -1}):
        with pytest.raises(ValueError):
            TreeSetRouting(phases=BOTH_PHASES, **fields)  # type: ignore[arg-type]
    rule = TreeSetRouting(phases=BOTH_PHASES, threshold=0.2, max_size=40, forced_top=2)
    assert rule.routes_blocks and rule.init_state(NUM_EXPERTS, jnp.float32) is None
    with pytest.raises(TypeError, match="no single-token step"):
        rule.route(jnp.zeros((1, NUM_EXPERTS)), jnp.ones((1,), dtype=bool), None, SoftmaxRouting(), NUM_ACTIVE)
    assert RoutingIntervention.from_json(rule.to_json()) == rule
