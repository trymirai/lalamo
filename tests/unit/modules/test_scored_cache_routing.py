"""Tests for `ScoredCacheRouting`: Cache-Prior re-ranking with a scored cache and its four extensions.

Expectations come from three places, never from the output of the rule itself: the already tested
`CacheConditionalRouting`, which the rule must equal with every extension off; a plain-Python reference of
the docstring of the rule (sets, dicts and sorted lists; float32 only where the rule rounds); and invariants
that hold for any cache (it never exceeds its capacity, it holds what was just dispatched).
"""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.mlp import (
    MixtureOfExperts,
    MLPForwardPassConfig,
    RoutingBias,
    RoutingFunction,
    RoutingIntervention,
    RoutingPhase,
    SigmoidRouting,
    SoftmaxRouting,
)
from lalamo.modules.routing_interventions import (
    CacheConditionalRouting,
    CachePriority,
    ExpertCacheState,
    ScoredCacheRouting,
    ScoredCacheState,
)
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.modules.test_moe_routing_trace import MODEL_DIM, _moe, _rng_array

NUM_EXPERTS = 16
NUM_ACTIVE = 4
CAPACITY = 6
BOTH_PHASES = (RoutingPhase.CONTEXT, RoutingPhase.GENERATION)
WEIGHT_SCALE = 8.0
GATED = SigmoidRouting(bias_mode=RoutingBias.GATING, weight_scale=WEIGHT_SCALE)
PLAIN = ScoredCacheRouting(phases=BOTH_PHASES, bias=0.6, forced_top=1, cache_capacity=CAPACITY)
EXTENSIONS = {
    "router-preference": replace(PLAIN, priority=CachePriority.ROUTER_PREFERENCE),
    "admission": replace(PLAIN, admission_budget=1, forced_top=0),
    "admission-of-two": replace(PLAIN, admission_budget=2, priority=CachePriority.ROUTER_PREFERENCE, forced_top=0),
    "graded": replace(PLAIN, graded_bonus=True, bias=1.0),
    "graded-by-preference": replace(PLAIN, graded_bonus=True, priority=CachePriority.ROUTER_PREFERENCE),
    "pruning": replace(PLAIN, prune_below_rank=2),
    "everything": replace(
        PLAIN,
        priority=CachePriority.ROUTER_PREFERENCE,
        admission_budget=1,
        graded_bonus=True,
        prune_below_rank=2,
        bias=1.0,
    ),
}
ROUTING_FUNCTIONS = [pytest.param(SoftmaxRouting(), id="softmax"), pytest.param(GATED, id="sigmoid-gating-bias")]


def _logits(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _biases(routing_function: RoutingFunction) -> np.ndarray | None:
    if isinstance(routing_function, SoftmaxRouting):
        return None
    return (0.2 * np.random.default_rng(11).normal(size=NUM_EXPERTS)).astype(np.float32)


def _state(rule: RoutingIntervention, batch: int, num_experts: int = NUM_EXPERTS) -> ScoredCacheState:
    template = rule.init_state(num_experts, jnp.float32)
    return jax.tree.map(lambda array: jnp.repeat(array[None, ...], batch, axis=0), template)


def _step(
    rule: RoutingIntervention,
    routing_function: RoutingFunction,
    logits: np.ndarray,
    active: np.ndarray,
    state: ScoredCacheState | ExpertCacheState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ScoredCacheState | ExpertCacheState]:
    """(indices, weights, dispatched slots, effective logits, updated state) of one routed token per row."""
    biases = _biases(routing_function)
    routing, updated = rule.route(
        jnp.asarray(logits),
        jnp.asarray(active),
        state,
        routing_function,
        NUM_ACTIVE,
        None if biases is None else jnp.asarray(biases),
    )
    assert routing.effective_logits is not None
    indices = np.asarray(routing.active_expert_indices)
    dispatched = np.ones(indices.shape, dtype=bool) if routing.dispatched is None else np.asarray(routing.dispatched)
    return (
        indices,
        np.asarray(routing.active_expert_weights),
        dispatched,
        np.asarray(routing.effective_logits),
        updated,
    )


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_with_every_extension_off_the_rule_is_cache_conditional_routing(routing_function: RoutingFunction) -> None:
    # Neutral element stated in the docstring: defaults give CacheConditionalRouting, token by token. The old
    # rule derives its cache from recency stamps and the new one keeps it as a set, so equal caches also check
    # the eviction of the set. Catches a wrong victim, a stale priority, a bonus on the wrong set.
    old = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.6, forced_top=1, cache_capacity=CAPACITY)
    batch, steps = 3, 24
    logits = _logits((steps, batch, NUM_EXPERTS), seed=1)
    active = np.random.default_rng(2).random((steps, batch)) < 0.7
    old_state, new_state = _state(old, batch), _state(PLAIN, batch)
    for step in range(steps):
        old_indices, old_weights, _, old_effective, old_state = _step(
            old, routing_function, logits[step], active[step], old_state
        )
        new_indices, new_weights, new_dispatched, new_effective, new_state = _step(
            PLAIN, routing_function, logits[step], active[step], new_state
        )
        assert isinstance(old_state, ExpertCacheState) and isinstance(new_state, ScoredCacheState)
        assert new_dispatched.all()
        np.testing.assert_array_equal(new_indices, old_indices)
        np.testing.assert_array_equal(new_weights, old_weights)
        np.testing.assert_array_equal(new_effective, old_effective)
        np.testing.assert_array_equal(np.asarray(new_state.last_used), np.asarray(old_state.last_used))
        stamps = np.asarray(old_state.last_used)
        for row in range(batch):
            most_recent = [int(expert) for expert in np.argsort(-stamps[row]) if stamps[row, expert] >= 0]
            assert set(np.flatnonzero(np.asarray(new_state.resident)[row]).tolist()) == set(most_recent[:CAPACITY])


class Reference:
    """The rule of the docstring for one sequence of one layer, in plain Python.

    Independent of the implementation: the cache is a set, the eviction order a sorted list, the selection a
    sort of expert ids; no array library ranks, masks or gathers anything. float32 is used only for the
    numbers the rule itself keeps in float32 (scores, boost, moving averages), so that a near-tie resolves
    the same way on both sides.
    """

    def __init__(self, rule: ScoredCacheRouting, routing_function: RoutingFunction, biases: np.ndarray | None) -> None:
        self.rule = rule
        self.routing_function = routing_function
        self.biases = biases
        self.resident: set[int] = set()
        self.last_used: dict[int, int] = {}
        self.fast = [np.float32(0.0)] * NUM_EXPERTS
        self.slow = [np.float32(0.0)] * NUM_EXPERTS
        self.clock = 0
        self.ranges: list[float] = []

    def _priority(self, expert: int) -> float:
        if self.rule.priority is CachePriority.RECENCY:
            return float(self.last_used.get(expert, -1))
        return float(self.fast[expert] + self.slow[expert])

    def _eviction_order(self, experts: set[int], priority: dict[int, float]) -> list[int]:
        return sorted(experts, key=lambda expert: (priority[expert], expert))

    def _insert(self, incoming: set[int], shielded: set[int], priority: dict[int, float]) -> None:
        overflow = max(len(self.resident | incoming) - CAPACITY, 0)
        victims = self._eviction_order(self.resident - shielded, priority)[:overflow]
        self.resident = (self.resident - set(victims)) | incoming

    def _scores(self, logits: np.ndarray) -> tuple[list[np.float32], list[float]]:
        """(scores the experts are ranked by, scores the weights are taken from)."""
        if isinstance(self.routing_function, SoftmaxRouting):
            return [np.float32(value) for value in logits], [float(value) for value in logits]
        assert self.biases is not None
        sigmoid = [1.0 / (1.0 + math.exp(-float(value))) for value in logits]
        gating = [np.float32(np.float32(score) + bias) for score, bias in zip(sigmoid, self.biases, strict=True)]
        return gating, sigmoid

    def _weights(self, weighting: list[float], selected: list[int]) -> list[float]:
        if isinstance(self.routing_function, SoftmaxRouting):
            peak = max(weighting[expert] for expert in selected)
            exponents = [math.exp(weighting[expert] - peak) for expert in selected]
            return [value / sum(exponents) for value in exponents]
        total = sum(weighting[expert] for expert in selected) + 1e-6
        return [weighting[expert] / total * WEIGHT_SCALE for expert in selected]

    def step(self, logits: np.ndarray, active: bool) -> tuple[list[int], dict[int, float]]:
        """The experts dispatched on this token and their weights."""
        rule = self.rule
        gating, weighting = self._scores(logits)
        token_range = float(max(gating) - min(gating))
        delta = np.float32(np.mean(self.ranges) if self.ranges else token_range)
        priority = {expert: self._priority(expert) for expert in range(NUM_EXPERTS)}
        ranking = sorted(range(NUM_EXPERTS), key=lambda expert: -gating[expert])
        own = ranking[:NUM_ACTIVE]

        missing = [expert for expert in own if expert not in self.resident][: rule.admission_budget]
        if missing:
            self._insert(set(missing), set(own), priority)

        forced = set(ranking[: rule.forced_top])
        if active:
            order = self._eviction_order(self.resident, priority)
            grade = {expert: np.float32(place + 1) / np.float32(len(order)) for place, expert in enumerate(order)}
            boosted = []
            for expert in range(NUM_EXPERTS):
                if expert in forced:
                    prior = np.float32(1.0)
                elif expert in self.resident:
                    prior = grade[expert] if rule.graded_bonus else np.float32(1.0)
                else:
                    prior = np.float32(0.0)
                boosted.append(gating[expert] + np.float32(rule.bias) * delta * prior)
            selected = sorted(range(NUM_EXPERTS), key=lambda expert: -boosted[expert])[:NUM_ACTIVE]
            kept = [
                expert
                for rank, expert in enumerate(selected)
                if rule.prune_below_rank is None
                or expert in self.resident
                or expert in forced
                or rank < rule.prune_below_rank
            ]
        else:
            selected = own
            kept = own
        full = dict(zip(selected, self._weights(weighting, selected), strict=True))
        scale = sum(full.values()) / sum(full[expert] for expert in kept)
        weights = {expert: full[expert] * scale for expert in kept}

        self._insert(set(kept), set(kept), priority)
        # Higher weight = smaller stamp = leaves first among the experts of this token.
        for rank, expert in enumerate(sorted(kept, key=lambda expert: -weights[expert])):
            self.last_used[expert] = self.clock * NUM_ACTIVE + rank
        for expert in range(NUM_EXPERTS):
            preferred = np.float32(expert in own)
            self.fast[expert] = np.float32(rule.FAST_DECAY) * self.fast[expert] + (
                np.float32(1.0 - rule.FAST_DECAY) * preferred
            )
            self.slow[expert] = np.float32(rule.SLOW_DECAY) * self.slow[expert] + (
                np.float32(1.0 - rule.SLOW_DECAY) * preferred
            )
        self.clock += 1
        self.ranges.append(token_range)
        return kept, weights


def _dispatched(indices: np.ndarray, weights: np.ndarray, dispatched: np.ndarray) -> dict[int, float]:
    """Expert -> weight over the dispatched slots of one token; the indices of a token must be distinct."""
    assert len(set(indices.tolist())) == len(indices)
    return {
        int(expert): float(weight)
        for expert, weight, kept in zip(indices.tolist(), weights.tolist(), dispatched.tolist(), strict=True)
        if kept
    }


def _trajectory(
    rule: ScoredCacheRouting, routing_function: RoutingFunction, logits: np.ndarray, active: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, ScoredCacheState]]:
    state = _state(rule, logits.shape[1])
    steps = []
    for step in range(logits.shape[0]):
        indices, weights, dispatched, _, state = _step(rule, routing_function, logits[step], active[step], state)
        assert isinstance(state, ScoredCacheState)
        steps.append((indices, weights, dispatched, state))
    return steps


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("name", EXTENSIONS)
def test_matches_the_python_reference_token_by_token(name: str, routing_function: RoutingFunction) -> None:
    # Differential test against the docstring: 40 tokens, three sequences, a mixed phase mask. The dispatched
    # experts, their weights and the cache must agree after every token. Catches an eviction from the wrong
    # pool, a grade taken from the wrong order, a dropped expert that still weighs or still enters the cache.
    rule = EXTENSIONS[name]
    batch, steps = 3, 40
    logits = _logits((steps, batch, NUM_EXPERTS), seed=3)
    active = np.random.default_rng(4).random((steps, batch)) < 0.7
    references = [Reference(rule, routing_function, _biases(routing_function)) for _ in range(batch)]
    trajectory = _trajectory(rule, routing_function, logits, active)
    for step, (indices, weights, dispatched, state) in enumerate(trajectory):
        for row in range(batch):
            kept, expected = references[row].step(logits[step, row], bool(active[step, row]))
            actual = _dispatched(indices[row], weights[row], dispatched[row])
            assert set(actual) == set(kept), (step, row)
            # A dropped expert must not weigh: its slot is computed by the dispatch like any other.
            assert (weights[row][~dispatched[row]] == 0.0).all(), (step, row)
            # float32 against Python floats: mantissa ~1.2e-7 on weights of magnitude up to WEIGHT_SCALE.
            np.testing.assert_allclose(
                [actual[expert] for expert in kept], [expected[expert] for expert in kept], atol=WEIGHT_SCALE * 1e-6
            )
            resident = set(np.flatnonzero(np.asarray(state.resident)[row]).tolist())
            assert resident == references[row].resident, (step, row)


@pytest.mark.parametrize("name", EXTENSIONS)
def test_every_extension_changes_the_routing_of_the_fixture(name: str) -> None:
    # Guard against a vacuous differential test: on the fixture each configuration must dispatch differently
    # from the rule with every extension off at the same bias and floor.
    rule = EXTENSIONS[name]
    plain = replace(PLAIN, bias=rule.bias, forced_top=rule.forced_top)
    logits = _logits((40, 3, NUM_EXPERTS), seed=3)
    active = np.random.default_rng(4).random((40, 3)) < 0.7
    extended = _trajectory(rule, SoftmaxRouting(), logits, active)
    reference = _trajectory(plain, SoftmaxRouting(), logits, active)
    assert any(
        not np.array_equal(np.sort(ours[0], axis=-1), np.sort(theirs[0], axis=-1))
        or not np.array_equal(ours[2], theirs[2])
        for ours, theirs in zip(extended, reference, strict=True)
    )


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
@pytest.mark.parametrize("name", EXTENSIONS)
def test_the_cache_never_overflows_and_holds_what_was_just_dispatched(
    name: str, routing_function: RoutingFunction
) -> None:
    # Invariants of any cache of `capacity` slots that is updated with the dispatched experts.
    rule = EXTENSIONS[name]
    logits = _logits((60, 4, NUM_EXPERTS), seed=5)
    active = np.random.default_rng(6).random((60, 4)) < 0.7
    for indices, weights, dispatched, state in _trajectory(rule, routing_function, logits, active):
        resident = np.asarray(state.resident)
        assert (resident.sum(axis=-1) <= CAPACITY).all()
        for row in range(indices.shape[0]):
            kept = set(_dispatched(indices[row], weights[row], dispatched[row]))
            assert kept <= set(np.flatnonzero(resident[row]).tolist())


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_pruning_keeps_the_total_weight_of_the_routing_function(routing_function: RoutingFunction) -> None:
    # Docstring: the dispatched experts are rescaled to the sum the routing function gave the whole selection
    # (1 under softmax, the weight scale under scaled sigmoid), a dropped slot weighs exactly zero, and the
    # first `prune_below_rank` ranks are always dispatched.
    rule = EXTENSIONS["pruning"]
    assert rule.prune_below_rank is not None
    logits = _logits((40, 3, NUM_EXPERTS), seed=7)
    active = np.ones((40, 3), dtype=bool)
    expected_total = 1.0 if isinstance(routing_function, SoftmaxRouting) else WEIGHT_SCALE
    for _, weights, dispatched, _ in _trajectory(rule, routing_function, logits, active):
        np.testing.assert_allclose(weights.sum(axis=-1), expected_total, rtol=1e-5)
        assert (weights[~dispatched] == 0.0).all()
        assert dispatched[:, : rule.prune_below_rank].all()


def test_admitted_experts_are_dispatched_on_the_token_that_admits_them() -> None:
    # Derived from the rule: an admitted expert is in the router's own top-k, gets the full bonus, and only
    # experts the router ranks above it can stay ahead of it, so with a binary bonus it is always selected.
    # Hence a token whose own top-k is not fully resident dispatches at least one expert new to the cache.
    rule = EXTENSIONS["admission"]
    logits = _logits((40, 3, NUM_EXPERTS), seed=8)
    active = np.ones((40, 3), dtype=bool)
    previous = np.zeros((3, NUM_EXPERTS), dtype=bool)
    for step, (indices, _, _, state) in enumerate(_trajectory(rule, SoftmaxRouting(), logits, active)):
        own = np.argsort(-logits[step], axis=-1, kind="stable")[:, :NUM_ACTIVE]
        for row in range(3):
            missing = [int(expert) for expert in own[row] if not previous[row, expert]]
            if missing:
                assert missing[0] in indices[row].tolist(), (step, row)
        previous = np.asarray(state.resident)


@pytest.mark.parametrize("routing_function", ROUTING_FUNCTIONS)
def test_a_cold_start_rule_routes_the_active_tokens_as_if_the_others_never_happened(
    routing_function: RoutingFunction,
) -> None:
    # Spec of `cold_start`: the state follows only the active tokens. So the active tokens of a sequence are
    # routed exactly as the rule without the flag routes them when they are the whole sequence, and the
    # inactive ones get the base routing. Catches a state that leaks from context tokens into generation.
    rule = replace(PLAIN, cold_start=True)
    steps = 30
    logits = _logits((steps, 1, NUM_EXPERTS), seed=14)
    active = np.random.default_rng(15).random((steps, 1)) < 0.6
    cold = _trajectory(rule, routing_function, logits, active)
    kept = np.flatnonzero(active[:, 0])
    alone = _trajectory(PLAIN, routing_function, logits[kept], np.ones((len(kept), 1), dtype=bool))
    for (indices, weights, _, state), (expected_indices, expected_weights, _, expected_state) in zip(
        [cold[step] for step in kept], alone, strict=True
    ):
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(weights, expected_weights)
        np.testing.assert_array_equal(np.asarray(state.last_used), np.asarray(expected_state.last_used))
    biases = _biases(routing_function)
    for step in np.flatnonzero(~active[:, 0]):
        base = routing_function(
            jnp.asarray(logits[step]), None if biases is None else jnp.asarray(biases), num_active=NUM_ACTIVE
        )
        np.testing.assert_array_equal(cold[step][0], np.asarray(base.active_expert_indices))


def test_only_a_pruning_rule_declares_that_it_may_drop_experts() -> None:
    # Contract with consumers of routing traces: indices of a token repeat only under a rule that says so.
    assert [name for name, rule in EXTENSIONS.items() if rule.may_drop_experts] == ["pruning", "everything"]
    assert not PLAIN.may_drop_experts
    assert not CacheConditionalRouting(phases=BOTH_PHASES, bias=0.5).may_drop_experts


def test_the_rule_survives_a_json_round_trip() -> None:
    # Candidate specs are JSON files: every field, the enum included, must come back as it was written.
    rule = EXTENSIONS["everything"]
    assert RoutingIntervention.from_json(rule.to_json()) == rule


@pytest.mark.parametrize(
    "arguments",
    [{"admission_budget": -1}, {"prune_below_rank": 0}, {"bias": 1.5}, {"forced_top": -1}, {"cache_capacity": 0}],
)
def test_invalid_settings_are_refused(arguments: dict[str, int | float]) -> None:
    with pytest.raises(ValueError, match="must"):
        replace(PLAIN, **arguments)


def test_a_cache_smaller_than_one_selection_is_refused() -> None:
    # The cache is updated with every dispatched expert, so it must be able to hold the experts of a token.
    rule = replace(PLAIN, cache_capacity=NUM_ACTIVE - 1)
    with pytest.raises(ValueError, match="cannot hold"):
        _step(rule, SoftmaxRouting(), _logits((2, NUM_EXPERTS), seed=9), np.ones(2, dtype=bool), _state(rule, 2))


def _module(rule: ScoredCacheRouting) -> MixtureOfExperts:
    base = _moe()
    experts = base.config.num_routed_experts
    biases = (0.2 * np.random.default_rng(12).normal(size=experts)).astype(np.float32)
    config = replace(base.config, routing_function=GATED, routing_intervention=rule)
    return replace(base, config=config, expert_bias=jnp.asarray(biases))


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize("name", ["router-preference", "everything", "cold-start"])
def test_prefill_and_token_by_token_decode_dispatch_alike(name: str) -> None:
    # MR (causality of the rule): a parallel prefill makes the routing decisions that decode steps make while
    # carrying the state. Runs the rule inside the mixture, under a mesh, through both of its dispatch paths.
    template = _moe().config
    # "cold-start" is the plain rule with the flag: the state must also be held back under a mesh.
    extension = replace(PLAIN, cold_start=True) if name == "cold-start" else EXTENSIONS[name]
    rule = replace(
        extension,
        cache_capacity=template.num_active_routed_experts + 1,
        prune_below_rank=None if extension.prune_below_rank is None else 1,
    )
    module = _module(rule)
    experts = template.num_routed_experts
    batch, tokens = 2, 8
    inputs = jax.device_put(
        _rng_array((batch, tokens, MODEL_DIM), seed=13), make_sharding((LogicalAxis.BATCH, None, None))
    )
    keychain = Keychain.init(7, sharding_config=make_test_sharding_config())

    prefill = module.call_with_routing(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.5),
        routing_state=_state(rule, batch, experts),
        keychain=keychain,
    )
    state = _state(rule, batch, experts)
    decoded_indices, decoded_outputs = [], []
    for position in range(tokens):
        step = module.call_with_routing(
            inputs[:, position : position + 1],
            forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.SINGLE_TOKEN),
            routing_state=state,
            keychain=keychain,
        )
        assert step.routing_trace is not None
        state = step.updated_routing_state
        decoded_indices.append(np.asarray(step.routing_trace.active_expert_indices))
        decoded_outputs.append(np.asarray(step.outputs))

    assert prefill.routing_trace is not None
    np.testing.assert_array_equal(
        np.sort(np.asarray(prefill.routing_trace.active_expert_indices), axis=-1),
        np.sort(np.concatenate(decoded_indices, axis=1), axis=-1),
    )
    # Two float32 dispatches of the same sums. An output is a sum of terms as large as the largest output
    # (about 200 on this fixture, the weights sum to 8), so the two orders may differ by a few ulps of that
    # scale: float32 mantissa ~1.2e-7, eight ulps allowed.
    decoded = np.concatenate(decoded_outputs, axis=1)
    np.testing.assert_allclose(np.asarray(prefill.outputs), decoded, rtol=0.0, atol=1e-6 * np.abs(decoded).max())
