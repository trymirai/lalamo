"""Tests for CacheConditionalRouting (Mixture of Cache-Conditional Experts, arXiv:2412.00099, Eq. 9-10).

Expectations are derived from the paper's definition -- boosted logits select, original logits weigh, an LRU
cache of `capacity` experts updated with the dispatched set -- and from a pure-Python reference of that
definition (lists and sets, float32 arithmetic where the implementation rounds), never from the code under
test. lambda = 0 is the neutral element and must reproduce the base routing bit for bit.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.mlp import (
    MixtureOfExperts,
    MLPForwardPassConfig,
    RoutingFunction,
    RoutingIntervention,
    RoutingMap,
    RoutingPhase,
    SoftmaxRouting,
)
from lalamo.modules.routing_interventions import CacheConditionalRouting, ExpertCacheState
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.modules.test_moe_routing_trace import MODEL_DIM, MOE_MODES, _moe, _rng_array

NUM_EXPERTS = 8
NUM_ACTIVE = 3
BOTH_PHASES = (RoutingPhase.CONTEXT, RoutingPhase.GENERATION)


def _state(intervention: CacheConditionalRouting, batch: int, num_experts: int = NUM_EXPERTS) -> ExpertCacheState:
    template = intervention.init_state(num_experts, jnp.float32)
    assert isinstance(template, ExpertCacheState)
    return jax.tree.map(lambda array: jnp.repeat(array[None, ...], batch, axis=0), template)


def _route(
    intervention: CacheConditionalRouting,
    logits: np.ndarray,
    active: np.ndarray,
    state: ExpertCacheState,
    num_active: int = NUM_ACTIVE,
) -> tuple[np.ndarray, np.ndarray, ExpertCacheState]:
    routing, updated = intervention.route(
        jnp.asarray(logits, dtype=jnp.float32),
        jnp.asarray(active),
        state,
        SoftmaxRouting(),
        num_active,
    )
    assert isinstance(updated, ExpertCacheState)
    return np.asarray(routing.active_expert_indices), np.asarray(routing.active_expert_weights), updated


def _cache_of(state: ExpertCacheState, row: int, capacity: int) -> set[int]:
    stamps = np.asarray(state.last_used)[row]
    used = [int(expert) for expert in np.argsort(-stamps) if stamps[expert] >= 0]
    return set(used[:capacity])


class ReferenceCache:
    """The paper's per-layer LRU in plain Python: a most-recent-first list of expert ids."""

    def __init__(self) -> None:
        self.order: list[int] = []
        self.ranges: list[float] = []

    def cached(self, capacity: int) -> set[int]:
        return set(self.order[:capacity])

    def step(
        self,
        logits: np.ndarray,
        active: bool,
        *,
        bias: float,
        forced_top: int,
        capacity: int,
        num_active: int,
        logit_range: float | None,
    ) -> tuple[list[int], list[float]]:
        logits32 = logits.astype(np.float32)
        token_range = float(logits32.max() - logits32.min())
        delta = (
            np.float32(logit_range)
            if logit_range is not None
            else np.float32(np.mean(self.ranges) if self.ranges else token_range)
        )
        ranking = sorted(range(len(logits32)), key=lambda expert: -logits32[expert])
        if active:
            boosted_set = self.cached(capacity) | set(ranking[:forced_top])
            boosted = logits32 + np.float32(bias) * delta * np.asarray(
                [expert in boosted_set for expert in range(len(logits32))], dtype=np.float32
            )
            selected = sorted(range(len(logits32)), key=lambda expert: -boosted[expert])[:num_active]
        else:
            selected = ranking[:num_active]
        chosen = np.asarray([logits32[expert] for expert in selected], dtype=np.float64)
        weights = np.exp(chosen - chosen.max())
        weights /= weights.sum()
        # Higher weight = evicted first, i.e. LEAST recent among the experts of this token.
        newest_first = sorted(selected, key=lambda expert: weights[selected.index(expert)])
        self.order = newest_first + [expert for expert in self.order if expert not in selected]
        self.ranges.append(token_range)
        return selected, weights.tolist()


@pytest.mark.parametrize("mode", MOE_MODES)
@pytest.mark.usefixtures("fake_mesh")
def test_zero_bias_reproduces_the_base_routing_bit_exact(mode: ForwardPassMode) -> None:
    # Neutral element (Eq. 9 with lambda = 0): the boosted logits are the logits, the selection and the softmax
    # weights are those of SoftmaxRouting, so outputs must be identical, not merely close.
    base = _moe()
    module = replace(
        base, config=replace(base.config, routing_intervention=CacheConditionalRouting(phases=BOTH_PHASES, bias=0.0))
    )
    tokens = 6 if mode == ForwardPassMode.MULTI_TOKEN else 1
    inputs = jax.device_put(_rng_array((2, tokens, MODEL_DIM), seed=3), make_sharding((LogicalAxis.BATCH, None, None)))
    config = MLPForwardPassConfig(mode=mode, moe_chunk_size_ratio=0.5)
    keychain = Keychain.init(7, sharding_config=make_test_sharding_config())

    reference = base(inputs, forward_pass_config=config, keychain=keychain)
    result = module(
        inputs,
        forward_pass_config=config,
        routing_state=_state(module.config.routing_intervention, 2),
        keychain=keychain,
    )

    assert reference.routing_trace is not None and result.routing_trace is not None
    np.testing.assert_array_equal(np.asarray(result.outputs), np.asarray(reference.outputs))
    np.testing.assert_array_equal(
        np.asarray(result.routing_trace.active_expert_indices),
        np.asarray(reference.routing_trace.active_expert_indices),
    )


def test_matches_the_python_lru_reference_over_a_sequence() -> None:
    # Differential test: twelve tokens, two sequences, a mixed phase mask; selection, weights and the final
    # cache contents must match the list-and-set reference of the paper's rule at every step.
    intervention = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.7, forced_top=1, cache_capacity=3)
    rng = np.random.default_rng(0)
    batch, steps = 2, 12
    logits = rng.normal(size=(steps, batch, NUM_EXPERTS)).astype(np.float32)
    active = rng.random((steps, batch)) < 0.6
    references = [ReferenceCache() for _ in range(batch)]
    state = _state(intervention, batch)
    for step in range(steps):
        indices, weights, state = _route(intervention, logits[step], active[step], state)
        for row in range(batch):
            expected_indices, expected_weights = references[row].step(
                logits[step, row],
                bool(active[step, row]),
                bias=0.7,
                forced_top=1,
                capacity=3,
                num_active=NUM_ACTIVE,
                logit_range=None,
            )
            assert indices[row].tolist() == expected_indices, (step, row)
            # fp32 softmax against a float64 reference: fp32 mantissa ~1.2e-7, weights sum to one.
            np.testing.assert_allclose(weights[row], expected_weights, atol=1e-6)
            assert _cache_of(state, row, capacity=3) == references[row].cached(3), (step, row)


def test_forced_top_experts_are_selected_even_against_a_full_bias() -> None:
    # Spec: the top-J experts of the original ranking are in the boosted mask too, so at lambda = 1 with a
    # cache full of other experts they still make the cut.
    intervention = CacheConditionalRouting(phases=BOTH_PHASES, bias=1.0, forced_top=2, cache_capacity=3)
    logits = np.asarray([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]], dtype=np.float32)
    state = _state(intervention, 1)
    # Make experts 5, 6, 7 the resident ones.
    state = replace(
        state, last_used=jnp.asarray([[-1, -1, -1, -1, -1, 2, 1, 0]], dtype=jnp.int32), clock=jnp.asarray([1])
    )

    indices, _, _ = _route(intervention, logits, np.asarray([True]), state)

    assert {0, 1} <= set(indices[0].tolist())
    assert len(set(indices[0].tolist()) & {5, 6, 7}) == 1  # the remaining slot goes to the cache


def test_bias_pulls_a_cached_expert_past_a_narrowly_better_uncached_one() -> None:
    # Construction: expert 3 is rank k+1 by 0.1 nats behind expert 2; expert 3 is cached, expert 2 is not.
    # The logit range is 7, so any lambda above 0.1 / 7 promotes expert 3 (Eq. 9); lambda = 0 does not.
    logits = np.asarray([[7.0, 6.0, 1.0, 0.9, 0.5, 0.2, 0.1, 0.0]], dtype=np.float32)

    def cache_state(intervention: CacheConditionalRouting) -> ExpertCacheState:
        return replace(
            _state(intervention, 1),
            last_used=jnp.asarray([[-1, -1, -1, 0, -1, -1, -1, -1]], dtype=jnp.int32),
            clock=jnp.asarray([1]),
        )

    pulled = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.5, forced_top=0, cache_capacity=4)
    untouched = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.0, forced_top=0, cache_capacity=4)

    pulled_indices, _, _ = _route(pulled, logits, np.asarray([True]), cache_state(pulled))
    untouched_indices, _, _ = _route(untouched, logits, np.asarray([True]), cache_state(untouched))

    assert set(pulled_indices[0].tolist()) == {0, 1, 3}
    assert set(untouched_indices[0].tolist()) == {0, 1, 2}


def test_weights_are_the_softmax_of_the_original_logits_over_the_selected_set() -> None:
    # Spec (paper §3.3): z' re-ranks only; the mixing weights come from the unmodified logits.
    intervention = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.9, forced_top=0, cache_capacity=4)
    logits = np.asarray([[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5]], dtype=np.float32)
    state = replace(
        _state(intervention, 1),
        last_used=jnp.asarray([[-1, -1, -1, -1, 3, 2, 1, 0]], dtype=jnp.int32),
        clock=jnp.asarray([1]),
    )

    indices, weights, _ = _route(intervention, logits, np.asarray([True]), state)

    chosen = logits[0, indices[0]].astype(np.float64)
    expected = np.exp(chosen - chosen.max())
    expected /= expected.sum()
    np.testing.assert_allclose(weights[0], expected, atol=1e-6)
    assert set(indices[0].tolist()) != {0, 1, 2}  # the bias did re-rank, so the check is not vacuous


def test_inactive_tokens_keep_the_base_routing_but_still_update_the_cache() -> None:
    # Architecture rule: the cache mirrors what was dispatched on every token; only the bias is phase-gated.
    intervention = CacheConditionalRouting(phases=(RoutingPhase.GENERATION,), bias=1.0, forced_top=0, cache_capacity=2)
    logits = np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
    state = _state(intervention, 1)

    indices, _, state = _route(intervention, logits, np.asarray([False]), state)

    assert set(indices[0].tolist()) == {7, 6, 5}
    assert int(np.asarray(state.clock)[0]) == 1
    assert _cache_of(state, 0, capacity=2) == {5, 6}  # expert 7 has the highest weight: evicted first


def test_running_logit_range_is_the_mean_over_the_tokens_seen() -> None:
    # Delta_avg (Eq. 10) as a running estimate: after T tokens the state holds sum and count of the ranges.
    intervention = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.5)
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5, 2, NUM_EXPERTS)).astype(np.float32)
    state = _state(intervention, 2)
    for step in range(5):
        _, _, state = _route(intervention, logits[step], np.asarray([True, False]), state)
    expected = (logits.max(axis=-1) - logits.min(axis=-1)).mean(axis=0)
    np.testing.assert_allclose(np.asarray(state.range_sum) / np.asarray(state.range_count), expected, rtol=1e-6)
    assert np.asarray(state.range_count).tolist() == [5, 5]


@pytest.mark.parametrize("replicated_experts", [False, True], ids=["chunked-dispatch", "ragged-dispatch"])
@pytest.mark.usefixtures("fake_mesh")
def test_prefill_and_decode_route_identically_through_the_moe(replicated_experts: bool) -> None:
    # The stateful rule inside the parallel prefill must equal the token-by-token decode, cache and all.
    base = _moe(replicated_experts=replicated_experts)
    intervention = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.8, forced_top=1, cache_capacity=3)
    module = replace(base, config=replace(base.config, routing_intervention=intervention))
    inputs = jax.device_put(_rng_array((2, 6, MODEL_DIM), seed=5), make_sharding((LogicalAxis.BATCH, None, None)))
    keychain = Keychain.init(7, sharding_config=make_test_sharding_config())

    prefill = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.5),
        routing_state=_state(intervention, 2),
        keychain=keychain,
    )
    state = _state(intervention, 2)
    stepped = []
    for position in range(6):
        step = module(
            inputs[:, position : position + 1],
            forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.SINGLE_TOKEN),
            routing_state=state,
            keychain=keychain,
        )
        assert step.routing_trace is not None
        stepped.append(np.asarray(step.routing_trace.active_expert_indices))
        state = step.updated_routing_state

    assert prefill.routing_trace is not None
    np.testing.assert_array_equal(
        np.asarray(prefill.routing_trace.active_expert_indices), np.concatenate(stepped, axis=1)
    )
    assert isinstance(prefill.updated_routing_state, ExpertCacheState) and isinstance(state, ExpertCacheState)
    np.testing.assert_array_equal(np.asarray(prefill.updated_routing_state.last_used), np.asarray(state.last_used))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32], ids=["bfloat16", "float32"])
def test_weights_keep_the_dtype_of_the_router_logits(dtype: jnp.dtype) -> None:
    # Regression: the intervention computed the bias in float32 and returned float32 weights, which turned a
    # bfloat16 MoE output into float32 and tripped the dtype assertion of TransformerLayer on the real model.
    # The mixing weights must be exactly what SoftmaxRouting would have produced on the same selected set.
    intervention = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.0, forced_top=0)
    logits = jnp.asarray(np.linspace(-2.0, 2.0, NUM_EXPERTS)[None, :], dtype=dtype)
    routing, _ = intervention.route(
        logits, jnp.ones((1,), dtype=jnp.bool_), _state(intervention, 1), SoftmaxRouting(), NUM_ACTIVE
    )
    base = SoftmaxRouting()(logits, NUM_ACTIVE)

    assert routing.active_expert_weights.dtype == logits.dtype
    np.testing.assert_array_equal(np.asarray(routing.active_expert_weights), np.asarray(base.active_expert_weights))
    np.testing.assert_array_equal(np.asarray(routing.active_expert_indices), np.asarray(base.active_expert_indices))


@pytest.mark.usefixtures("fake_mesh")
def test_a_bfloat16_mixture_keeps_its_dtype_through_the_intervention() -> None:
    # The same regression at the module level, where the failure actually surfaced: a bf16 model must come out
    # of the MoE in bf16, with the intervention active and re-ranking.
    module = _moe().astype(jnp.bfloat16)
    module = replace(
        module,
        config=replace(module.config, routing_intervention=CacheConditionalRouting(phases=BOTH_PHASES, bias=0.7)),
    )
    inputs = jax.device_put(
        _rng_array((2, 4, MODEL_DIM), seed=11).astype(jnp.bfloat16), make_sharding((LogicalAxis.BATCH, None, None))
    )

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.5),
        routing_state=_state(module.config.routing_intervention, 2),
        keychain=Keychain.init(7, sharding_config=make_test_sharding_config()),
    )

    assert result.outputs.dtype == inputs.dtype


def test_spec_round_trips_validates_and_defaults_the_capacity_to_half_the_experts() -> None:
    spec = CacheConditionalRouting(phases=(RoutingPhase.GENERATION,), bias=0.5, forced_top=2)
    assert RoutingIntervention.from_json(spec.to_json()) == spec
    assert spec.capacity(256) == 128
    assert CacheConditionalRouting(phases=(), bias=0.1, cache_capacity=5).capacity(8) == 5
    with pytest.raises(ValueError, match="bias"):
        CacheConditionalRouting(phases=(), bias=1.5)
    with pytest.raises(ValueError, match="exceeds"):
        CacheConditionalRouting(phases=(), bias=0.5, cache_capacity=9).capacity(8)
    with pytest.raises(TypeError, match="SoftmaxRouting"):
        spec.route(jnp.zeros((1, 8)), jnp.ones((1,), dtype=jnp.bool_), _state(spec, 1), _NotSoftmax(), NUM_ACTIVE)


@dataclass(frozen=True)
class _NotSoftmax(RoutingFunction):
    def call_unbatched(self, logits: jax.Array, num_active: int) -> RoutingMap:
        _, indices = jax.lax.top_k(logits, num_active)
        return RoutingMap(
            active_expert_indices=indices, active_expert_weights=jnp.full((num_active,), 1.0 / num_active)
        )


def _unused(module: MixtureOfExperts) -> None:  # pragma: no cover - keeps the import meaningful for pyrefly
    del module
