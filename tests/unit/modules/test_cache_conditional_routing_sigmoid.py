"""Tests for `CacheConditionalRouting` under routing functions other than the paper's softmax.

The rule is defined in the scores a routing function takes its top-k over, so every expectation here is
derived from two things only: the definition of the routing function (sigmoid scores, an optional per-expert
bias on the gating side, weights normalised over the selection and scaled) and the paper's rule applied to
those scores. Nothing is read off the implementation's own output.
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
    SigmoidRouting,
    SoftmaxRouting,
)
from lalamo.modules.routing_interventions import CacheConditionalRouting, ExpertCacheState
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.modules.test_cache_conditional_routing import BOTH_PHASES, NUM_ACTIVE, NUM_EXPERTS, _cache_of, _state
from tests.unit.modules.test_moe_routing_trace import MODEL_DIM, _moe, _rng_array

WEIGHT_SCALE = 8.0
GATED = SigmoidRouting(bias_mode=RoutingBias.GATING, weight_scale=WEIGHT_SCALE)
RULE = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.7, forced_top=1, cache_capacity=3)
ROUTING_FUNCTIONS = [
    pytest.param(SoftmaxRouting(), False, id="softmax"),
    pytest.param(SigmoidRouting(), False, id="sigmoid"),
    pytest.param(GATED, True, id="sigmoid-gating-bias"),
    pytest.param(SigmoidRouting(bias_mode=RoutingBias.WEIGHTS, weight_scale=2.0), True, id="sigmoid-weights-bias"),
    pytest.param(SigmoidRouting(bias_mode=RoutingBias.BOTH, weight_scale=2.0), True, id="sigmoid-both-bias"),
]


def _logits(shape: tuple[int, ...], seed: int) -> np.ndarray:
    # Standard normal logits keep the sigmoid scores away from saturation, where float32 would merge them.
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _biases(seed: int) -> np.ndarray:
    return (0.2 * np.random.default_rng(seed).normal(size=NUM_EXPERTS)).astype(np.float32)


@pytest.mark.parametrize(("routing_function", "has_biases"), ROUTING_FUNCTIONS)
def test_gating_scores_and_selection_weights_reproduce_the_routing_function(
    routing_function: RoutingFunction, has_biases: bool
) -> None:
    # Spec of the two methods: the top-k of `gating_scores` is the function's own selection, and `weights_of`
    # that selection are the function's own weights. A rule built on them is then the base routing whenever
    # it leaves the scores alone.
    logits = jnp.asarray(_logits((5, NUM_EXPERTS), seed=1))
    biases = jnp.asarray(_biases(seed=2)) if has_biases else None

    base = routing_function(logits, biases, num_active=NUM_ACTIVE)
    _, selected = jax.lax.top_k(routing_function.gating_scores(logits, biases), NUM_ACTIVE)

    np.testing.assert_array_equal(np.asarray(selected), np.asarray(base.active_expert_indices))
    # Same float32 arithmetic on both sides; one ulp of slack for the batched and the per-row evaluation order.
    np.testing.assert_allclose(
        np.asarray(routing_function.weights_of(logits, selected, biases)),
        np.asarray(base.active_expert_weights),
        rtol=3e-7,
    )


def _route(
    intervention: CacheConditionalRouting,
    logits: np.ndarray,
    active: np.ndarray,
    state: ExpertCacheState,
    biases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, ExpertCacheState]:
    routing, updated = intervention.route(
        jnp.asarray(logits), jnp.asarray(active), state, GATED, NUM_ACTIVE, jnp.asarray(biases)
    )
    assert isinstance(updated, ExpertCacheState)
    return np.asarray(routing.active_expert_indices), np.asarray(routing.active_expert_weights), updated


def test_zero_bias_is_the_base_sigmoid_routing_at_every_step() -> None:
    # Neutral element (Eq. 9 with lambda = 0): the boosted scores are the scores, whatever the cache holds.
    intervention = replace(RULE, bias=0.0)
    logits, biases = _logits((6, 2, NUM_EXPERTS), seed=3), _biases(seed=4)
    state = _state(intervention, 2)
    for step in range(logits.shape[0]):
        indices, weights, state = _route(intervention, logits[step], np.ones(2, dtype=bool), state, biases)
        base = GATED(jnp.asarray(logits[step]), jnp.asarray(biases), num_active=NUM_ACTIVE)
        np.testing.assert_array_equal(indices, np.asarray(base.active_expert_indices))
        np.testing.assert_array_equal(weights, np.asarray(base.active_expert_weights))


class SigmoidReferenceCache:
    """The paper's per-layer LRU over sigmoid routing, in plain Python floats.

    Independent of the implementation: scores come from `math.exp`, the cache is a most-recent-first list,
    and no array library takes part in the selection.
    """

    def __init__(self) -> None:
        self.order: list[int] = []
        self.ranges: list[float] = []

    def cached(self, capacity: int) -> set[int]:
        return set(self.order[:capacity])

    def step(self, logits: np.ndarray, biases: np.ndarray, active: bool) -> tuple[list[int], list[float]]:
        scores = [1.0 / (1.0 + math.exp(-float(logit))) for logit in logits]
        gating = [score + float(bias) for score, bias in zip(scores, biases, strict=True)]
        token_range = max(gating) - min(gating)
        delta = sum(self.ranges) / len(self.ranges) if self.ranges else token_range
        ranking = sorted(range(len(gating)), key=lambda expert: -gating[expert])
        if active:
            boosted_set = self.cached(RULE.cache_capacity or 0) | set(ranking[: RULE.forced_top])
            boosted = [value + RULE.bias * delta * (expert in boosted_set) for expert, value in enumerate(gating)]
            selected = sorted(range(len(boosted)), key=lambda expert: -boosted[expert])[:NUM_ACTIVE]
        else:
            selected = ranking[:NUM_ACTIVE]
        total = sum(scores[expert] for expert in selected) + 1e-6
        weights = [scores[expert] / total * WEIGHT_SCALE for expert in selected]
        # Higher weight = evicted first, i.e. LEAST recent among the experts of this token.
        newest_first = sorted(selected, key=lambda expert: weights[selected.index(expert)])
        self.order = newest_first + [expert for expert in self.order if expert not in selected]
        self.ranges.append(token_range)
        return selected, weights


def test_matches_the_python_lru_reference_over_a_sequence_of_sigmoid_routed_tokens() -> None:
    # Differential test: twelve tokens, two sequences, a mixed phase mask. Selection, weights and the cache
    # contents must match the reference at every step. Catches a rule applied to the raw logits instead of
    # the gating scores, a dropped gating bias, softmax weights left in place of the scaled sigmoid ones.
    batch, steps = 2, 12
    logits, biases = _logits((steps, batch, NUM_EXPERTS), seed=5), _biases(seed=6)
    active = np.random.default_rng(7).random((steps, batch)) < 0.6
    references = [SigmoidReferenceCache() for _ in range(batch)]
    state = _state(RULE, batch)
    for step in range(steps):
        indices, weights, state = _route(RULE, logits[step], active[step], state, biases)
        for row in range(batch):
            expected_indices, expected_weights = references[row].step(
                logits[step, row], biases, bool(active[step, row])
            )
            assert indices[row].tolist() == expected_indices, (step, row)
            # float32 against Python floats: mantissa ~1.2e-7, weights of magnitude up to WEIGHT_SCALE.
            np.testing.assert_allclose(weights[row], expected_weights, atol=WEIGHT_SCALE * 1e-6)
            assert _cache_of(state, row, capacity=3) == references[row].cached(3), (step, row)


def _sigmoid_moe() -> MixtureOfExperts:
    base = _moe()
    config = replace(base.config, routing_function=GATED, routing_intervention=RULE)
    return replace(base, config=config, expert_bias=jnp.asarray(_biases(seed=8)))


@pytest.mark.usefixtures("fake_mesh")
def test_prefill_and_token_by_token_decode_route_alike_under_the_rule_with_a_gating_bias() -> None:
    # MR (causality of the rule): a parallel prefill over six tokens must make the routing decisions that six
    # decode steps make while carrying the state. Catches a gating bias that reaches only one of the two
    # dispatches of the mixture.
    module = _sigmoid_moe()
    batch, tokens = 2, 6
    inputs = jax.device_put(
        _rng_array((batch, tokens, MODEL_DIM), seed=9), make_sharding((LogicalAxis.BATCH, None, None))
    )
    keychain = Keychain.init(7, sharding_config=make_test_sharding_config())

    prefill = module.call_with_routing(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.5),
        routing_state=_state(RULE, batch),
        keychain=keychain,
    )
    state = _state(RULE, batch)
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
    # Two float32 dispatches of the same sums: allow a few ulps of an output of order one.
    np.testing.assert_allclose(np.asarray(prefill.outputs), np.concatenate(decoded_outputs, axis=1), atol=1e-5)


@pytest.mark.usefixtures("fake_mesh")
def test_the_rule_redirects_routing_of_the_sigmoid_fixture() -> None:
    # Guard against a vacuous pass of the tests above: with lambda = 0.7 and a cache of three of eight experts
    # the rule must change at least one routing decision on this input.
    module = _sigmoid_moe()
    base = replace(module, config=replace(module.config, routing_intervention=None))
    inputs = jax.device_put(_rng_array((2, 6, MODEL_DIM), seed=9), make_sharding((LogicalAxis.BATCH, None, None)))
    config = MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.5)
    keychain = Keychain.init(7, sharding_config=make_test_sharding_config())

    ruled = module.call_with_routing(
        inputs, forward_pass_config=config, routing_state=_state(RULE, 2), keychain=keychain
    )
    plain = base.call_with_routing(inputs, forward_pass_config=config, keychain=keychain)

    assert ruled.routing_trace is not None and plain.routing_trace is not None
    assert not np.array_equal(
        np.sort(np.asarray(ruled.routing_trace.active_expert_indices), axis=-1),
        np.sort(np.asarray(plain.routing_trace.active_expert_indices), axis=-1),
    )
