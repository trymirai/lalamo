"""Tests for `WindowedRouting`: a rule restarted at the start of every window of active tokens.

Expectations come from the docstring of the wrapper and from the rule it wraps, which has tests of its own:
inside a window the wrapped rule runs as on a sequence that begins with the window. So every expected value
here is produced by the wrapped rule alone, started from its initial state where the docstring says a window
opens, or by the model's own routing where the rule has no history to act on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.module import ForwardPassMode
from lalamo.modules.mlp import RoutingFunction, RoutingIntervention, RoutingPhase
from lalamo.modules.routing_interventions import (
    CacheConditionalRouting,
    ScoredCacheRouting,
    WindowedRouting,
    WindowState,
)
from tests.unit.modules.test_moe_routing_intervention import _batched_state, _config, _get, _inputs, _keychain, _with
from tests.unit.modules.test_moe_routing_trace import _moe
from tests.unit.modules.test_scored_cache_routing import (
    BOTH_PHASES,
    CAPACITY,
    NUM_ACTIVE,
    NUM_EXPERTS,
    ROUTING_FUNCTIONS,
    _biases,
    _logits,
    _state,
    _step,
)

WINDOW = 5
CACHE = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.6, forced_top=1, cache_capacity=CAPACITY)
SCORED = ScoredCacheRouting(phases=BOTH_PHASES, bias=0.6, forced_top=1, cache_capacity=CAPACITY, graded_bonus=True)
INNER_RULES = [pytest.param(CACHE, id="cache"), pytest.param(SCORED, id="scored-cache")]


def windowed(rule: RoutingIntervention, window: int = WINDOW) -> WindowedRouting:
    return WindowedRouting(phases=rule.phases, rule=rule, window=window)


def run(
    rule: RoutingIntervention, routing_function: RoutingFunction, logits: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(indices, weights, effective logits) as [steps, batch, ...] of a rule run from its initial state."""
    state = _state(rule, logits.shape[1])
    indices, weights, effective = [], [], []
    for step in range(logits.shape[0]):
        step_indices, step_weights, _, step_effective, state = _step(
            rule, routing_function, logits[step], active[step], state
        )
        indices.append(step_indices)
        weights.append(step_weights)
        effective.append(step_effective)
    return np.stack(indices), np.stack(weights), np.stack(effective)


def restarted_reference(
    rule: RoutingIntervention, routing_function: RoutingFunction, logits: np.ndarray, active: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The docstring of the wrapper, one sequence at a time and in plain control flow.

    Independent of the wrapper: no state of the wrapper and no masked selection are involved. A sequence is
    walked token by token with the wrapped rule alone; whenever the token is active and the number of active
    tokens seen before it is a multiple of the window, the rule is handed its initial state.
    """
    steps, batch, _ = logits.shape
    indices = np.zeros((steps, batch, NUM_ACTIVE), dtype=np.int64)
    weights = np.zeros((steps, batch, NUM_ACTIVE), dtype=np.float32)
    effective = np.zeros(logits.shape, dtype=np.float32)
    for row in range(batch):
        state = _state(rule, 1)
        seen = 0
        for step in range(steps):
            if active[step, row] and seen % window == 0:
                state = _state(rule, 1)
            step_indices, step_weights, _, step_effective, state = _step(
                rule, routing_function, logits[step, row : row + 1], active[step, row : row + 1], state
            )
            indices[step, row], weights[step, row], effective[step, row] = (
                step_indices[0],
                step_weights[0],
                step_effective[0],
            )
            seen += int(active[step, row])
    return indices, weights, effective


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("rule", INNER_RULES)
def test_every_window_is_the_rule_on_a_sequence_that_begins_with_the_window(
    rule: RoutingIntervention, routing_function: RoutingFunction
) -> None:
    # Docstring: with every token active the sequence is cut into consecutive windows, and each is routed as
    # a sequence of its own. So the run over three windows and a part of a fourth is the concatenation of
    # four runs of the rule alone. Catches a state that survives a window, a window of the wrong length, a
    # restart that comes one token late.
    steps, batch = 3 * WINDOW + 2, 3
    logits = _logits((steps, batch, NUM_EXPERTS), seed=3)
    active = np.ones((steps, batch), dtype=bool)
    got = run(windowed(rule), routing_function, logits, active)
    pieces = [
        run(rule, routing_function, logits[start : start + WINDOW], active[start : start + WINDOW])
        for start in range(0, steps, WINDOW)
    ]
    for field in range(3):
        np.testing.assert_array_equal(got[field], np.concatenate([piece[field] for piece in pieces]))
    # The fixture is not vacuous: the rule alone, never restarted, routes these tokens differently.
    assert not np.array_equal(got[0], run(rule, routing_function, logits, active)[0])


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("rule", INNER_RULES)
def test_inactive_tokens_belong_to_no_window_and_every_sequence_counts_its_own(
    rule: RoutingIntervention, routing_function: RoutingFunction
) -> None:
    # Docstring: windows are cut over the active tokens of a sequence; the tokens in between are routed as
    # inactive ones and do not move the window. The reference walks every sequence on its own, so a batch
    # whose rows have different masks also checks that rows do not share a window counter.
    steps, batch = 31, 4
    logits = _logits((steps, batch, NUM_EXPERTS), seed=5)
    active = np.random.default_rng(6).random((steps, batch)) < 0.6
    active[:3, 0] = False  # a sequence that opens with inactive tokens: its first window starts later
    got = run(windowed(rule), routing_function, logits, active)
    want = restarted_reference(rule, routing_function, logits, active, WINDOW)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[2], want[2])


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_the_first_token_of_every_window_is_routed_by_the_model_itself(routing_function: RoutingFunction) -> None:
    # Docstring, last sentence: a cache rule starts from an empty history, so at the first token of a window
    # nothing is resident and the forced experts are the router's own -- the selection is the routing
    # function's own top-k with its own weights. Elsewhere the rule does change the routing.
    steps, batch = 4 * WINDOW, 3
    logits = _logits((steps, batch, NUM_EXPERTS), seed=7)
    active = np.ones((steps, batch), dtype=bool)
    indices, weights, _ = run(windowed(CACHE), routing_function, logits, active)
    biases = _biases(routing_function)
    changed_elsewhere = 0
    for step in range(steps):
        base = routing_function(
            jnp.asarray(logits[step]), None if biases is None else jnp.asarray(biases), num_active=NUM_ACTIVE
        )
        base_indices, base_weights = np.asarray(base.active_expert_indices), np.asarray(base.active_expert_weights)
        same_sets = np.array_equal(np.sort(indices[step], axis=-1), np.sort(base_indices, axis=-1))
        if step % WINDOW == 0:
            assert same_sets
            order, base_order = np.argsort(indices[step], axis=-1), np.argsort(base_indices, axis=-1)
            np.testing.assert_array_equal(
                np.take_along_axis(weights[step], order, axis=-1),
                np.take_along_axis(base_weights, base_order, axis=-1),
            )
        else:
            changed_elsewhere += not same_sets
    assert changed_elsewhere > 0


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("rule", INNER_RULES)
def test_a_window_that_holds_the_whole_sequence_is_the_rule_itself(
    rule: RoutingIntervention, routing_function: RoutingFunction
) -> None:
    # Neutral element: with every token active and a window no shorter than the sequence there is a single
    # window that opens at the first token, where the rule has its initial state anyway.
    steps, batch = 17, 2
    logits = _logits((steps, batch, NUM_EXPERTS), seed=9)
    active = np.ones((steps, batch), dtype=bool)
    got = run(windowed(rule, window=steps), routing_function, logits, active)
    want = run(rule, routing_function, logits, active)
    for field in range(3):
        np.testing.assert_array_equal(got[field], want[field])


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_a_window_of_one_token_is_the_base_routing(routing_function: RoutingFunction) -> None:
    # Degenerate limit with a known answer: every active token opens a window, the cache rule never has a
    # history, so no token leaves the routing function's own top-k.
    steps, batch = 12, 2
    logits = _logits((steps, batch, NUM_EXPERTS), seed=10)
    active = np.random.default_rng(11).random((steps, batch)) < 0.7
    indices, _, _ = run(windowed(CACHE, window=1), routing_function, logits, active)
    biases = _biases(routing_function)
    for step in range(steps):
        base = routing_function(
            jnp.asarray(logits[step]), None if biases is None else jnp.asarray(biases), num_active=NUM_ACTIVE
        )
        np.testing.assert_array_equal(
            np.sort(indices[step], axis=-1), np.sort(np.asarray(base.active_expert_indices), axis=-1)
        )


def test_the_window_counter_advances_by_one_per_active_token() -> None:
    # Spec of the state: `active_tokens` is the number of active tokens the sequence has seen.
    rule = windowed(CACHE)
    steps, batch = 9, 3
    logits = _logits((steps, batch, NUM_EXPERTS), seed=12)
    active = np.random.default_rng(13).random((steps, batch)) < 0.5
    state = _state(rule, batch)
    for step in range(steps):
        *_, state = _step(rule, ROUTING_FUNCTIONS[0].values[0], logits[step], active[step], state)
    assert isinstance(state, WindowState)
    np.testing.assert_array_equal(np.asarray(state.active_tokens), active.sum(axis=0))


def test_a_window_is_cut_over_the_phases_of_its_rule_and_has_a_positive_length() -> None:
    with pytest.raises(ValueError, match="phases of its rule"):
        WindowedRouting(phases=(RoutingPhase.GENERATION,), rule=CACHE, window=4)
    with pytest.raises(ValueError, match="window must be positive"):
        windowed(CACHE, window=0)


def test_the_wrapper_round_trips_through_json_with_its_rule() -> None:
    rule = windowed(SCORED, window=16)
    restored = RoutingIntervention.from_json(rule.to_json())
    assert restored == rule
    assert restored.digest() == rule.digest()
    assert windowed(SCORED, window=8).digest() != rule.digest()


@pytest.mark.parametrize("replicated_experts", [False, True], ids=["chunked-dispatch", "ragged-dispatch"])
@pytest.mark.usefixtures("fake_mesh")
def test_prefill_and_decode_route_a_windowed_rule_identically(replicated_experts: bool) -> None:
    # The claim every causal rule must keep (see test_moe_routing_intervention.py): inside the parallel
    # prefill a token is routed exactly as the token-by-token decode routes it, with the same final state.
    # Here the state includes the window counter, and the windows open at different tokens in the two rows.
    inner = CacheConditionalRouting(phases=(RoutingPhase.GENERATION,), bias=0.8, forced_top=1, cache_capacity=4)
    module = _with(_moe(replicated_experts=replicated_experts), windowed(inner, window=3))
    tokens = 11
    inputs = _inputs(batch=2, tokens=tokens)
    mask = jnp.asarray([[position >= 1 for position in range(tokens)], [position >= 4 for position in range(tokens)]])

    prefill = module.call_with_routing(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        generation_mask=mask,
        routing_state=_batched_state(module, batch=2),
        keychain=_keychain(),
    )
    state = _batched_state(module, batch=2)
    stepped = []
    for position in range(tokens):
        step = module.call_with_routing(
            inputs[:, position : position + 1],
            forward_pass_config=_config(ForwardPassMode.SINGLE_TOKEN),
            generation_mask=mask[:, position : position + 1],
            routing_state=state,
            keychain=_keychain(),
        )
        assert step.routing_trace is not None
        stepped.append(_get(step.routing_trace.active_expert_indices))
        state = step.updated_routing_state

    assert prefill.routing_trace is not None
    np.testing.assert_array_equal(_get(prefill.routing_trace.active_expert_indices), np.concatenate(stepped, axis=1))
    assert isinstance(prefill.updated_routing_state, WindowState) and isinstance(state, WindowState)
    np.testing.assert_array_equal(_get(prefill.updated_routing_state.active_tokens), np.asarray([10, 7]))
    for prefill_leaf, stepped_leaf in zip(
        jax.tree.leaves(prefill.updated_routing_state), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(_get(prefill_leaf), _get(stepped_leaf))
