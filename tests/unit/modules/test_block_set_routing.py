"""Tests for `BlockSetRouting`: one set of experts for every block of active tokens, chosen from the whole block.

Expectations come from the docstring of the rule, never from its output: a plain-Python reference walks one
sequence, cuts its active tokens into blocks, adds up the shares of the model's own mixture per expert with a
dictionary, picks the set by sorting, and ranks the experts of the set for every token by sorting. It uses the
routing function only for what the docstring delegates to it (the model's own routing, the ranking scores, the
weights of a selection), all of which have tests of their own. Besides the reference: invariants of any block
rule, neutral elements, and the symmetry that makes the rule non-causal.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.module import ForwardPassMode
from lalamo.modules.mlp import RoutingFunction, RoutingIntervention, RoutingPhase
from lalamo.modules.routing_interventions import BlockObjective, BlockSetRouting, BlockState
from tests.unit.modules.test_moe_routing_intervention import _batched_state, _config, _get, _inputs, _keychain, _with
from tests.unit.modules.test_moe_routing_trace import _moe
from tests.unit.modules.test_scored_cache_routing import (
    BOTH_PHASES,
    NUM_ACTIVE,
    NUM_EXPERTS,
    ROUTING_FUNCTIONS,
    _biases,
    _logits,
)

WINDOW = 5
SIZED = BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, size=7)
THRESHOLDED = BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, threshold=0.45)
RULES = [pytest.param(SIZED, id="size"), pytest.param(THRESHOLDED, id="threshold")]


def run(
    rule: BlockSetRouting,
    routing_function: RoutingFunction,
    logits: np.ndarray,
    active: np.ndarray,
    seen: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(indices, weights, effective logits, active tokens seen afterwards) of one call over [tokens, batch]."""
    batch = logits.shape[1]
    state = BlockState(active_tokens=jnp.zeros((batch,), dtype=jnp.int32) if seen is None else jnp.asarray(seen))
    biases = _biases(routing_function)
    routing, updated = rule.route_block(
        jnp.asarray(logits),
        jnp.asarray(active),
        state,
        routing_function,
        NUM_ACTIVE,
        None if biases is None else jnp.asarray(biases),
    )
    assert isinstance(updated, BlockState) and routing.effective_logits is not None
    return (
        np.asarray(routing.active_expert_indices),
        np.asarray(routing.active_expert_weights),
        np.asarray(routing.effective_logits),
        np.asarray(updated.active_tokens),
    )


def own_routing(routing_function: RoutingFunction, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(indices, weights, ranking scores) the routing function itself gives every row of [rows, experts]."""
    biases = _biases(routing_function)
    jax_biases = None if biases is None else jnp.asarray(biases)
    base = routing_function(jnp.asarray(logits), jax_biases, num_active=NUM_ACTIVE)
    scores = routing_function.gating_scores(jnp.asarray(logits), jax_biases)
    return np.asarray(base.active_expert_indices), np.asarray(base.active_expert_weights), np.asarray(scores)


def reference_sets(
    rule: BlockSetRouting, routing_function: RoutingFunction, logits: np.ndarray, active: np.ndarray, seen: int = 0
) -> list[set[int] | None]:
    """The set of the block of every token of one sequence (None for a token outside every block).

    Plain Python by the docstring: a counter of active tokens, a dictionary of masses per block, a sort.
    """
    indices, weights, _ = own_routing(routing_function, logits)
    block_of: list[int | None] = []
    masses: dict[int, dict[int, float]] = {}
    for token in range(logits.shape[0]):
        if not active[token]:
            block_of.append(None)
            continue
        block = seen // rule.window
        seen += 1
        block_of.append(block)
        total = float(np.sum(weights[token].astype(np.float64)))
        for expert, weight in zip(indices[token].tolist(), weights[token].tolist(), strict=True):
            masses.setdefault(block, {}).setdefault(expert, 0.0)
            masses[block][expert] += weight / total
    sets: dict[int, set[int]] = {}
    for block, mass in masses.items():
        ranked = sorted(range(NUM_EXPERTS), key=lambda expert: (-mass.get(expert, 0.0), expert))
        positive = sorted(mass.values())
        gaps = np.diff(np.asarray(positive))
        # The fixture must not hinge on a float32 tie between two masses: the rule adds them in float32.
        assert gaps.size == 0 or gaps.min() > 1e-5 or np.allclose(gaps.min(), 0.0, atol=0.0)
        if rule.size is not None:
            sets[block] = set(ranked[: rule.size])
        else:
            assert rule.threshold is not None
            chosen = {expert for expert, value in mass.items() if value >= rule.threshold}
            assert all(abs(value - rule.threshold) > 1e-5 for value in mass.values())
            sets[block] = chosen | set(ranked[:NUM_ACTIVE])
    return [None if block is None else sets[block] for block in block_of]


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("rule", RULES)
def test_every_token_is_routed_inside_the_set_of_its_block_as_the_docstring_says(
    rule: BlockSetRouting, routing_function: RoutingFunction
) -> None:
    # Differential test against the plain-Python reference: the set of every block, then for every token the
    # experts of the set in the order of the routing function's own scores (ties to the lower index) with the
    # weights the routing function gives that selection; tokens outside the phase keep the model's routing.
    tokens, batch = 33, 3
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=21)
    active = np.random.default_rng(22).random((tokens, batch)) < 0.75
    active[:4, 1] = False
    indices, weights, effective, _ = run(rule, routing_function, logits, active)
    biases = _biases(routing_function)
    jax_biases = None if biases is None else jnp.asarray(biases)
    restricted_somewhere = False
    for row in range(batch):
        sets = reference_sets(rule, routing_function, logits[:, row], active[:, row])
        own_indices, own_weights, scores = own_routing(routing_function, logits[:, row])
        for token in range(tokens):
            allowed = sets[token]
            if allowed is None:
                np.testing.assert_array_equal(indices[token, row], own_indices[token])
                np.testing.assert_array_equal(weights[token, row], own_weights[token])
                np.testing.assert_array_equal(effective[token, row], scores[token])
                continue
            ranked = sorted(allowed, key=lambda expert: (-float(scores[token, expert]), expert))[:NUM_ACTIVE]
            np.testing.assert_array_equal(indices[token, row], np.asarray(ranked))
            expected_weights = routing_function.weights_of(
                jnp.asarray(logits[token : token + 1, row]), jnp.asarray([ranked]), jax_biases
            )
            np.testing.assert_array_equal(weights[token, row], np.asarray(expected_weights)[0])
            outside = np.asarray([expert not in allowed for expert in range(NUM_EXPERTS)])
            assert np.isneginf(effective[token, row][outside]).all()
            np.testing.assert_array_equal(effective[token, row][~outside], scores[token][~outside])
            restricted_somewhere |= set(ranked) != set(own_indices[token].tolist())
    assert restricted_somewhere  # the fixture does change the routing


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_a_block_never_dispatches_more_experts_than_its_set_holds(routing_function: RoutingFunction) -> None:
    # Invariant the rule exists for: the experts of all the tokens of a block lie in one set of `size`
    # experts, and every token still gets its full number of distinct experts.
    tokens, batch = 4 * WINDOW, 4
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=23)
    active = np.ones((tokens, batch), dtype=bool)
    indices, _, _, _ = run(SIZED, routing_function, logits, active)
    for row in range(batch):
        for start in range(0, tokens, WINDOW):
            block = indices[start : start + WINDOW, row]
            assert len(set(block.reshape(-1).tolist())) <= SIZED.size
            assert all(len(set(token.tolist())) == NUM_ACTIVE for token in block)


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_the_order_of_the_tokens_inside_a_block_does_not_change_its_set(routing_function: RoutingFunction) -> None:
    # MR that tells the rule from a causal one: the set is a function of the block as a whole, so permuting
    # the tokens of every block permutes their routing with them and changes nothing else.
    tokens, batch = 3 * WINDOW, 2
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=24)
    active = np.ones((tokens, batch), dtype=bool)
    permutation = np.concatenate(
        [start + np.random.default_rng(start).permutation(WINDOW) for start in range(0, tokens, WINDOW)]
    )
    indices, weights, _, _ = run(SIZED, routing_function, logits, active)
    shuffled_indices, shuffled_weights, _, _ = run(SIZED, routing_function, logits[permutation], active)
    np.testing.assert_array_equal(shuffled_indices, indices[permutation])
    np.testing.assert_array_equal(shuffled_weights, weights[permutation])


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, size=NUM_EXPERTS), id="set-of-all-experts"),
        pytest.param(BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, threshold=0.0), id="threshold-zero"),
        pytest.param(BlockSetRouting(phases=BOTH_PHASES, window=1, size=NUM_ACTIVE), id="block-of-one-token"),
    ],
)
def test_a_set_that_holds_every_expert_a_token_wants_is_the_model_routing(
    rule: BlockSetRouting, routing_function: RoutingFunction
) -> None:
    # Neutral elements with a known answer: a set of all the experts, a threshold every expert passes, and a
    # block of one token whose set is exactly its own experts -- nothing is taken away from any token.
    tokens, batch = 13, 2
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=25)
    active = np.random.default_rng(26).random((tokens, batch)) < 0.8
    indices, weights, _, _ = run(rule, routing_function, logits, active)
    for row in range(batch):
        own_indices, own_weights, _ = own_routing(routing_function, logits[:, row])
        np.testing.assert_array_equal(indices[:, row], own_indices)
        np.testing.assert_array_equal(weights[:, row], own_weights)


@pytest.mark.parametrize("rule", RULES)
def test_a_call_that_ends_on_a_block_boundary_can_be_continued_in_the_next_call(rule: BlockSetRouting) -> None:
    # Docstring: blocks are counted over the active tokens of the sequence, across calls. Two calls split
    # at a block boundary of every row therefore give what one call gives; the state carries the count.
    routing_function = ROUTING_FUNCTIONS[1].values[0]
    tokens, batch, split = 4 * WINDOW, 2, 2 * WINDOW
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=27)
    active = np.ones((tokens, batch), dtype=bool)
    whole = run(rule, routing_function, logits, active)
    first = run(rule, routing_function, logits[:split], active[:split])
    second = run(rule, routing_function, logits[split:], active[split:], seen=first[3])
    np.testing.assert_array_equal(first[3], np.full(batch, split))
    np.testing.assert_array_equal(second[3], whole[3])
    for field in range(3):
        np.testing.assert_array_equal(np.concatenate([first[field], second[field]]), whole[field])


def test_a_block_cut_by_the_end_of_a_call_is_finished_on_its_own() -> None:
    # Docstring, last sentence: the part of a block that arrives in the next call is chosen from its own
    # tokens. With three tokens of a block already seen, the first two tokens of the call are the rest of
    # that block and form a set of their own; the tokens after them start whole blocks.
    routing_function = ROUTING_FUNCTIONS[0].values[0]
    tokens = 2 + 2 * WINDOW
    logits = _logits((tokens, 1, NUM_EXPERTS), seed=28)
    active = np.ones((tokens, 1), dtype=bool)
    indices, _, _, seen = run(SIZED, routing_function, logits, active, seen=np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(seen, [3 + tokens])
    sets = reference_sets(SIZED, routing_function, logits[:, 0], active[:, 0], seen=3)
    assert sets[0] == sets[1] and sets[1] != sets[2]
    for token in range(tokens):
        allowed = sets[token]
        assert allowed is not None and set(indices[token, 0].tolist()) <= allowed


def test_the_rule_has_no_single_token_step_and_validates_its_parameters() -> None:
    state = BlockState(active_tokens=jnp.zeros((1,), dtype=jnp.int32))
    with pytest.raises(TypeError, match="no single-token step"):
        SIZED.route(
            jnp.zeros((1, NUM_EXPERTS)), jnp.ones((1,), dtype=bool), state, ROUTING_FUNCTIONS[0].values[0], NUM_ACTIVE
        )
    with pytest.raises(ValueError, match="exactly one of size and threshold"):
        BlockSetRouting(phases=BOTH_PHASES, window=4)
    with pytest.raises(ValueError, match="exactly one of size and threshold"):
        BlockSetRouting(phases=BOTH_PHASES, window=4, size=8, threshold=0.5)
    with pytest.raises(ValueError, match="window must be positive"):
        BlockSetRouting(phases=BOTH_PHASES, window=0, size=8)
    too_small = BlockSetRouting(phases=BOTH_PHASES, window=4, size=NUM_ACTIVE - 1)
    with pytest.raises(ValueError, match="experts of one token"):
        run(
            too_small,
            ROUTING_FUNCTIONS[0].values[0],
            _logits((4, 1, NUM_EXPERTS), seed=1),
            np.ones((4, 1), dtype=bool),
        )


def test_the_rule_round_trips_through_json() -> None:
    for rule in (SIZED, THRESHOLDED):
        restored = RoutingIntervention.from_json(rule.to_json())
        assert restored == rule and restored.digest() == rule.digest()
    assert SIZED.digest() != THRESHOLDED.digest()


@pytest.mark.parametrize("replicated_experts", [False, True], ids=["chunked-dispatch", "ragged-dispatch"])
@pytest.mark.usefixtures("fake_mesh")
def test_a_mixture_dispatches_the_block_routing_in_a_multi_token_pass(replicated_experts: bool) -> None:
    # Inside a mixture of experts: the trace of a multi-token pass names, for the tokens of the generation
    # phase, experts that fit one set per block, and the tokens before it keep the routing of the same
    # mixture without the rule. A rule whose set holds every expert leaves the outputs bit-exact.
    window, size = 3, 5
    rule = BlockSetRouting(phases=(RoutingPhase.GENERATION,), window=window, size=size)
    plain = _moe(replicated_experts=replicated_experts)
    module = _with(plain, rule)
    tokens = 11
    inputs = _inputs(batch=2, tokens=tokens)
    mask = jnp.asarray([[position >= 2 for position in range(tokens)], [position >= 5 for position in range(tokens)]])

    def call(mixture, state):  # noqa: ANN001, ANN202
        return mixture.call_with_routing(
            inputs,
            forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
            generation_mask=mask,
            routing_state=state,
            keychain=_keychain(),
        )

    ruled, base = call(module, _batched_state(module, batch=2)), call(plain, None)
    assert ruled.routing_trace is not None and base.routing_trace is not None
    experts, base_experts = (
        _get(ruled.routing_trace.active_expert_indices),
        _get(base.routing_trace.active_expert_indices),
    )
    for row, first in enumerate((2, 5)):
        np.testing.assert_array_equal(experts[row, :first], base_experts[row, :first])
        for start in range(first, tokens, window):
            assert len(set(experts[row, start : start + window].reshape(-1).tolist())) <= size
    assert isinstance(ruled.updated_routing_state, BlockState)
    np.testing.assert_array_equal(_get(ruled.updated_routing_state.active_tokens), [tokens - 2, tokens - 5])

    everything = _with(
        plain, BlockSetRouting(phases=(RoutingPhase.GENERATION,), window=window, size=plain.config.mixture_size)
    )
    np.testing.assert_array_equal(
        _get(call(everything, _batched_state(everything, batch=2)).outputs), _get(base.outputs)
    )


@pytest.mark.usefixtures("fake_mesh")
def test_a_token_by_token_decode_refuses_a_block_rule() -> None:
    # Spec of `routes_blocks`: a rule that looks at the later tokens of a block cannot run one token at a
    # time; the mixture says so instead of quietly routing every token on its own.
    module = _with(_moe(), BlockSetRouting(phases=(RoutingPhase.GENERATION,), window=3, size=5))
    with pytest.raises(ValueError, match="token-by-token decode"):
        module.call_with_routing(
            _inputs(batch=2, tokens=1),
            forward_pass_config=_config(ForwardPassMode.SINGLE_TOKEN),
            generation_mask=jnp.ones((2, 1), dtype=bool),
            routing_state=_batched_state(module, batch=2),
            keychain=_keychain(),
        )


def test_block_state_counts_only_the_active_tokens() -> None:
    tokens, batch = 12, 3
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=29)
    active = np.random.default_rng(30).random((tokens, batch)) < 0.5
    _, _, _, seen = run(SIZED, ROUTING_FUNCTIONS[0].values[0], logits, active)
    np.testing.assert_array_equal(seen, active.sum(axis=0))
    assert jax.tree.leaves(SIZED.init_state(NUM_EXPERTS, jnp.float32))[0].shape == ()


# ------------------------------------------------------------------------------- the SCORE objective

SCORE_SIZED = BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, size=7, objective=BlockObjective.SCORE)
SCORE_THRESHOLDED = BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, threshold=0.12, objective=BlockObjective.SCORE)
SCORE_RULES = [pytest.param(SCORE_SIZED, id="size"), pytest.param(SCORE_THRESHOLDED, id="threshold")]


def reference_score_sets(
    rule: BlockSetRouting, routing_function: RoutingFunction, logits: np.ndarray, active: np.ndarray
) -> list[set[int] | None]:
    """The greedy of the docstring for one sequence, in plain Python: lists, sorts and a loop over experts.

    Values are computed in float64 from the float32 ranking scores, so every decision of the greedy is
    checked not to hinge on rounding: the best gain must lead the next one, and stand off the threshold, by
    far more than float32 can blur.
    """
    own_indices, _, scores = own_routing(routing_function, logits)
    values: dict[int, list[float]] = {}
    blocks: dict[int, list[int]] = {}
    block_of: list[int | None] = []
    seen = 0
    for token in range(logits.shape[0]):
        if not active[token]:
            block_of.append(None)
            continue
        lowest = float(scores[token].min())
        own_total = sum(float(scores[token, expert]) - lowest for expert in own_indices[token].tolist())
        values[token] = [(float(score) - lowest) / own_total for score in scores[token].tolist()]
        blocks.setdefault(seen // rule.window, []).append(token)
        block_of.append(seen // rule.window)
        seen += 1
    sets: dict[int, set[int]] = {}
    steps = rule.size if rule.size is not None else min(rule.max_size, NUM_EXPERTS)
    for block, tokens in blocks.items():
        chosen: list[int] = []
        for _ in range(steps):
            weakest = {}
            for token in tokens:
                kept = sorted((values[token][expert] for expert in chosen), reverse=True)[:NUM_ACTIVE]
                weakest[token] = kept[NUM_ACTIVE - 1] if len(kept) == NUM_ACTIVE else 0.0
            gains = {
                expert: sum(max(values[token][expert] - weakest[token], 0.0) for token in tokens)
                for expert in range(NUM_EXPERTS)
                if expert not in chosen
            }
            ranked = sorted(gains, key=lambda expert: (-gains[expert], expert))
            best = ranked[0]
            assert len(ranked) == 1 or gains[best] - gains[ranked[1]] > 1e-5 or gains[best] == gains[ranked[1]]
            if rule.threshold is not None and len(chosen) >= NUM_ACTIVE:
                assert abs(gains[best] - rule.threshold) > 1e-5
                if gains[best] < rule.threshold:
                    break
            chosen.append(best)
        sets[block] = set(chosen)
    return [None if block is None else sets[block] for block in block_of]


def sets_of(effective: np.ndarray, active: np.ndarray) -> list[set[int] | None]:
    """The set every active token of one sequence was routed in: the experts whose scores were not masked."""
    return [
        set(np.flatnonzero(np.isfinite(row)).tolist()) if flag else None
        for row, flag in zip(effective, active, strict=True)
    ]


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("rule", SCORE_RULES)
def test_the_score_objective_grows_the_set_by_the_greedy_of_the_docstring(
    rule: BlockSetRouting, routing_function: RoutingFunction
) -> None:
    # Differential test against the plain-Python greedy: the set of every block, and the routing of every
    # token inside it (the experts of the set in the order of the ranking scores, the model's routing for
    # the tokens outside the phase).
    tokens, batch = 28, 3
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=51)
    active = np.random.default_rng(42).random((tokens, batch)) < 0.8
    indices, _, effective, _ = run(rule, routing_function, logits, active)
    sizes = set()
    for row in range(batch):
        expected = reference_score_sets(rule, routing_function, logits[:, row], active[:, row])
        assert sets_of(effective[:, row], active[:, row]) == expected
        own_indices, _, scores = own_routing(routing_function, logits[:, row])
        for token, allowed in enumerate(expected):
            if allowed is None:
                np.testing.assert_array_equal(indices[token, row], own_indices[token])
                continue
            ranked = sorted(allowed, key=lambda expert: (-float(scores[token, expert]), expert))[:NUM_ACTIVE]
            np.testing.assert_array_equal(indices[token, row], np.asarray(ranked))
            sizes.add(len(allowed))
    if rule.threshold is not None:
        assert len(sizes) > 1  # the fixture has blocks of different sizes: the size does follow the block


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_a_larger_score_set_contains_the_smaller_one(routing_function: RoutingFunction) -> None:
    # MR of a greedy: the set of size m + 1 is the set of size m plus one expert, block by block.
    tokens, batch = 3 * WINDOW, 2
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=43)
    active = np.ones((tokens, batch), dtype=bool)
    previous = None
    for size in range(NUM_ACTIVE, NUM_ACTIVE + 5):
        rule = BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, size=size, objective=BlockObjective.SCORE)
        _, _, effective, _ = run(rule, routing_function, logits, active)
        current = [sets_of(effective[:, row], active[:, row]) for row in range(batch)]
        assert all(len(allowed) == size for row in current for allowed in row if allowed is not None)
        if previous is not None:
            assert all(
                old <= new
                for old_row, new_row in zip(previous, current, strict=True)
                for old, new in zip(old_row, new_row, strict=True)
            )
        previous = current


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_a_score_threshold_nothing_falls_below_is_the_model_routing(routing_function: RoutingFunction) -> None:
    # Neutral element: with threshold zero the greedy never stops before every expert is in, so no token
    # is denied any of its own experts.
    tokens, batch = 13, 2
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=44)
    active = np.random.default_rng(45).random((tokens, batch)) < 0.8
    rule = BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, threshold=0.0, objective=BlockObjective.SCORE)
    indices, weights, _, _ = run(rule, routing_function, logits, active)
    for row in range(batch):
        own_indices, own_weights, _ = own_routing(routing_function, logits[:, row])
        np.testing.assert_array_equal(indices[:, row], own_indices)
        np.testing.assert_array_equal(weights[:, row], own_weights)


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_a_score_set_is_never_smaller_than_one_token_needs_nor_larger_than_max_size(
    routing_function: RoutingFunction,
) -> None:
    # Docstring: a threshold nothing reaches still leaves the experts one token needs, and a threshold
    # everything reaches is cut at `max_size`. Every token keeps its full number of distinct experts.
    tokens, batch = 2 * WINDOW, 2
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=46)
    active = np.ones((tokens, batch), dtype=bool)
    for threshold, max_size, expected in ((1e6, 48, NUM_ACTIVE), (0.0, NUM_ACTIVE + 2, NUM_ACTIVE + 2)):
        rule = BlockSetRouting(
            phases=BOTH_PHASES, window=WINDOW, threshold=threshold, objective=BlockObjective.SCORE, max_size=max_size
        )
        indices, _, effective, _ = run(rule, routing_function, logits, active)
        for row in range(batch):
            assert all(len(allowed) == expected for allowed in sets_of(effective[:, row], active[:, row]))
        assert all(len(set(token.tolist())) == NUM_ACTIVE for token in indices.reshape(-1, NUM_ACTIVE))


def test_the_score_set_does_not_depend_on_the_order_of_the_tokens_of_a_block() -> None:
    # The same symmetry as for the mass objective: the greedy sums gains over the tokens of the block.
    routing_function = ROUTING_FUNCTIONS[1].values[0]
    tokens, batch = 3 * WINDOW, 2
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=47)
    active = np.ones((tokens, batch), dtype=bool)
    permutation = np.concatenate(
        [start + np.random.default_rng(start + 7).permutation(WINDOW) for start in range(0, tokens, WINDOW)]
    )
    indices, weights, _, _ = run(SCORE_THRESHOLDED, routing_function, logits, active)
    shuffled_indices, shuffled_weights, _, _ = run(SCORE_THRESHOLDED, routing_function, logits[permutation], active)
    np.testing.assert_array_equal(shuffled_indices, indices[permutation])
    np.testing.assert_array_equal(shuffled_weights, weights[permutation])


def test_the_objective_is_part_of_the_spec() -> None:
    restored = RoutingIntervention.from_json(SCORE_THRESHOLDED.to_json())
    assert restored == SCORE_THRESHOLDED
    assert SCORE_SIZED.digest() != SIZED.digest()
    with pytest.raises(ValueError, match="max_size must be positive"):
        BlockSetRouting(phases=BOTH_PHASES, window=4, threshold=0.1, objective=BlockObjective.SCORE, max_size=0)


@pytest.mark.usefixtures("fake_mesh")
def test_a_mixture_runs_the_score_objective_under_a_mesh() -> None:
    # The greedy carries arrays through a scan; inside a mixture they must agree in sharding with what the
    # logits give. A set that may hold every expert leaves the outputs of the mixture bit-exact.
    plain = _moe()
    rule = BlockSetRouting(phases=(RoutingPhase.GENERATION,), window=3, threshold=0.0, objective=BlockObjective.SCORE)
    module = _with(plain, rule)
    inputs = _inputs(batch=2, tokens=8)
    mask = jnp.ones((2, 8), dtype=bool)

    def call(mixture, state):  # noqa: ANN001, ANN202
        return mixture.call_with_routing(
            inputs,
            forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
            generation_mask=mask,
            routing_state=state,
            keychain=_keychain(),
        )

    np.testing.assert_array_equal(
        _get(call(module, _batched_state(module, batch=2)).outputs), _get(call(plain, None).outputs)
    )
    tight = _with(
        plain, BlockSetRouting(phases=(RoutingPhase.GENERATION,), window=3, size=5, objective=BlockObjective.SCORE)
    )
    trace = call(tight, _batched_state(tight, batch=2)).routing_trace
    assert trace is not None
    experts = _get(trace.active_expert_indices)
    assert all(
        len(set(experts[row, start : start + 3].reshape(-1).tolist())) <= 5 for row in range(2) for start in (0, 3)
    )


# ---------------------------------------------------------------------------------------- the floor

FLOORED = [
    pytest.param(BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, size=7, forced_top=1), id="mass-size"),
    pytest.param(
        BlockSetRouting(phases=BOTH_PHASES, window=WINDOW, threshold=0.45, forced_top=2), id="mass-threshold"
    ),
    pytest.param(
        BlockSetRouting(
            phases=BOTH_PHASES, window=WINDOW, threshold=0.2, objective=BlockObjective.SCORE, forced_top=1
        ),
        id="score-threshold",
    ),
    pytest.param(
        BlockSetRouting(
            phases=BOTH_PHASES, window=WINDOW, threshold=0.2, objective=BlockObjective.SCORE, forced_top=2
        ),
        id="score-threshold-two",
    ),
]


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("rule", FLOORED)
def test_the_first_ranked_experts_of_every_token_are_in_the_set_of_its_block(
    rule: BlockSetRouting, routing_function: RoutingFunction
) -> None:
    # Spec of `forced_top`: the experts the routing function ranks in the first `forced_top` places at any
    # token of a block are in the set of the block, so every token dispatches its own first ones. Without
    # the floor the same rule does drop some of them in this fixture.
    tokens, batch = 4 * WINDOW, 3
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=61)
    active = np.ones((tokens, batch), dtype=bool)
    indices, _, effective, _ = run(rule, routing_function, logits, active)
    bare = BlockSetRouting(
        phases=BOTH_PHASES, window=WINDOW, size=rule.size, threshold=rule.threshold, objective=rule.objective
    )
    bare_indices, _, _, _ = run(bare, routing_function, logits, active)
    dropped_without_floor = 0
    for row in range(batch):
        _, _, scores = own_routing(routing_function, logits[:, row])
        sets = sets_of(effective[:, row], active[:, row])
        for start in range(0, tokens, WINDOW):
            block = range(start, start + WINDOW)
            wanted = {
                expert
                for token in block
                for expert in sorted(range(NUM_EXPERTS), key=lambda e: (-float(scores[token, e]), e))[
                    : rule.forced_top
                ]
            }
            assert all(sets[token] is not None and wanted <= sets[token] for token in block)
        for token in range(tokens):
            first = sorted(range(NUM_EXPERTS), key=lambda e: (-float(scores[token, e]), e))[: rule.forced_top]
            np.testing.assert_array_equal(indices[token, row, : rule.forced_top], np.asarray(first))
            dropped_without_floor += not set(first) <= set(bare_indices[token, row].tolist())
    assert dropped_without_floor > 0


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("objective", list(BlockObjective))
def test_a_floor_of_every_expert_a_token_selects_is_the_model_routing(
    objective: BlockObjective, routing_function: RoutingFunction
) -> None:
    # Neutral element: forcing all the experts each token selects leaves nothing to take away, whatever the
    # threshold asks for.
    tokens, batch = 11, 2
    logits = _logits((tokens, batch, NUM_EXPERTS), seed=62)
    active = np.random.default_rng(63).random((tokens, batch)) < 0.8
    rule = BlockSetRouting(
        phases=BOTH_PHASES,
        window=WINDOW,
        threshold=1e6,
        objective=objective,
        forced_top=NUM_ACTIVE,
        max_size=NUM_EXPERTS,
    )
    indices, weights, _, _ = run(rule, routing_function, logits, active)
    for row in range(batch):
        own_indices, own_weights, _ = own_routing(routing_function, logits[:, row])
        np.testing.assert_array_equal(indices[:, row], own_indices)
        np.testing.assert_array_equal(weights[:, row], own_weights)


def test_the_score_greedy_goes_on_after_the_floor_as_it_would_from_that_set() -> None:
    # Docstring: under SCORE the floor is laid first and the greedy then continues. So the set with a floor
    # contains the floor, and what it adds beyond the floor are experts whose gain, given the floor, reaches
    # the threshold: raising the threshold out of reach leaves exactly the floor (when it already holds the
    # experts one token needs).
    routing_function = ROUTING_FUNCTIONS[0].values[0]
    tokens = 2 * WINDOW
    logits = _logits((tokens, 1, NUM_EXPERTS), seed=64)
    active = np.ones((tokens, 1), dtype=bool)
    _, _, scores = own_routing(routing_function, logits[:, 0])
    floor = 2
    only_floor = BlockSetRouting(
        phases=BOTH_PHASES, window=WINDOW, threshold=1e6, objective=BlockObjective.SCORE, forced_top=floor
    )
    _, _, effective, _ = run(only_floor, routing_function, logits, active)
    sets = sets_of(effective[:, 0], active[:, 0])
    for start in range(0, tokens, WINDOW):
        wanted = {
            expert
            for token in range(start, start + WINDOW)
            for expert in sorted(range(NUM_EXPERTS), key=lambda e: (-float(scores[token, e]), e))[:floor]
        }
        assert len(wanted) >= NUM_ACTIVE  # otherwise the minimum size would add experts beyond the floor
        assert sets[start] == wanted
    with pytest.raises(ValueError, match="forced_top must be non-negative"):
        BlockSetRouting(phases=BOTH_PHASES, window=4, size=8, forced_top=-1)
