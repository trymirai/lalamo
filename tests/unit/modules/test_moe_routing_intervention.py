"""Tests for phase-gated causal routing interventions: RoutingIntervention, generation masks, routing state.

Expectations come from the contract of the interface, never from the code under test: the identity
intervention must be invisible; a rule acts on exactly the tokens the phase mask marks active; a stateful
rule sees its sequence in time order, one token after the other, whatever the kernel path; the trace names
the experts the dispatch used. The interventions exercised here are test doubles defined below, so no
production rule is assumed to exist.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.mlp import (
    IdentityRoutingIntervention,
    MixtureOfExperts,
    MLPForwardPassConfig,
    RoutingFunction,
    RoutingIntervention,
    RoutingMap,
    RoutingPhase,
    with_routing_intervention,
)
from lalamo.modules.token_mixer import StateLayerBase, TransformerLayerState
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.modules.test_moe_routing_trace import (
    HIDDEN_DIM,
    MODEL_DIM,
    MOE_MODES,
    NUM_ACTIVE,
    _moe,
    _rng_array,
)

FIXED_EXPERTS = (5, 1, 6)  # in this order; random logits make an accidental match with the base top-k negligible


class CountState(StateLayerBase):
    count: Int[Array, "*batch"]


@dataclass(frozen=True)
class CountingIntervention(RoutingIntervention):
    """Base routing everywhere; the state counts the active tokens each sequence has seen."""

    def init_state(self, num_experts: int, dtype: DTypeLike) -> StateLayerBase | None:  # noqa: ARG002
        return CountState(count=jnp.zeros((), dtype=jnp.int32))

    def route(
        self,
        router_logits: Float[Array, "batch experts"],
        active: Bool[Array, " batch"],
        state: StateLayerBase | None,
        routing_function: RoutingFunction,
        num_active: int,
    ) -> tuple[RoutingMap, StateLayerBase | None]:
        assert isinstance(state, CountState)
        return routing_function(router_logits, num_active), CountState(count=state.count + active.astype(jnp.int32))


class SeenState(StateLayerBase):
    seen: Bool[Array, "*batch"]


@dataclass(frozen=True)
class SwitchIntervention(RoutingIntervention):
    """From the first active token on, route every token of the sequence to FIXED_EXPERTS with equal weights.

    The switch is sticky through the state, so the routing of a token depends on the past of its sequence:
    exactly the kind of rule whose prefill must be a causal scan.
    """

    def init_state(self, num_experts: int, dtype: DTypeLike) -> StateLayerBase | None:  # noqa: ARG002
        return SeenState(seen=jnp.zeros((), dtype=jnp.bool_))

    def route(
        self,
        router_logits: Float[Array, "batch experts"],
        active: Bool[Array, " batch"],
        state: StateLayerBase | None,
        routing_function: RoutingFunction,
        num_active: int,
    ) -> tuple[RoutingMap, StateLayerBase | None]:
        assert isinstance(state, SeenState)
        base = routing_function(router_logits, num_active)
        switched = state.seen | active
        batch_size = router_logits.shape[0]
        fixed_indices = jnp.broadcast_to(
            jnp.asarray(FIXED_EXPERTS, dtype=base.active_expert_indices.dtype), (batch_size, num_active)
        )
        fixed_weights = jnp.full((batch_size, num_active), 1.0 / num_active, dtype=base.active_expert_weights.dtype)
        routing = RoutingMap(
            active_expert_indices=jnp.where(switched[:, None], fixed_indices, base.active_expert_indices),
            active_expert_weights=jnp.where(switched[:, None], fixed_weights, base.active_expert_weights),
        )
        return routing, SeenState(seen=switched)


def _with(module: MixtureOfExperts, intervention: RoutingIntervention | None) -> MixtureOfExperts:
    return replace(module, config=replace(module.config, routing_intervention=intervention))


def _inputs(batch: int, tokens: int, seed: int = 10) -> Array:
    return jax.device_put(
        _rng_array((batch, tokens, MODEL_DIM), seed=seed), make_sharding((LogicalAxis.BATCH, None, None))
    )


def _keychain() -> Keychain:
    return Keychain.init(7, sharding_config=make_test_sharding_config())


def _config(mode: ForwardPassMode) -> MLPForwardPassConfig:
    return MLPForwardPassConfig(mode=mode, moe_chunk_size_ratio=0.5)


def _batched_state(module: MixtureOfExperts, batch: int) -> StateLayerBase | None:
    template = module.init_routing_state(jnp.float32)
    if template is None:
        return None
    return jax.tree.map(lambda array: jnp.repeat(array[None, ...], batch, axis=0), template)


def _mask(batch: int, tokens: int, active_from: int) -> Bool[Array, "batch tokens"]:
    return jnp.broadcast_to(jnp.arange(tokens) >= active_from, (batch, tokens))


def _get(array: Array) -> np.ndarray:
    return np.asarray(jax.device_get(array))


# ------------------------------------------------------------------------------------------ identity


@pytest.mark.parametrize("mode", MOE_MODES)
@pytest.mark.parametrize("replicated_experts", [False, True], ids=["chunked-dispatch", "ragged-dispatch"])
@pytest.mark.parametrize(
    "phases",
    [(), (RoutingPhase.CONTEXT, RoutingPhase.GENERATION)],
    ids=["inactive", "active-everywhere"],
)
@pytest.mark.usefixtures("fake_mesh")
def test_identity_intervention_leaves_outputs_and_routing_bit_exact(
    mode: ForwardPassMode,
    replicated_experts: bool,
    phases: tuple[RoutingPhase, ...],
) -> None:
    # Neutral element: identity routing through the intervention scan must equal the plain routing function
    # bit for bit -- the same top-k and softmax are applied to the same logits, only per time step.
    base = _moe(num_shared_experts=2, with_gate=True, replicated_experts=replicated_experts)
    module = _with(base, IdentityRoutingIntervention(phases=phases))
    inputs = _inputs(batch=2, tokens=6 if mode == ForwardPassMode.MULTI_TOKEN else 1)

    reference = base(inputs, forward_pass_config=_config(mode), keychain=_keychain())
    result = module(inputs, forward_pass_config=_config(mode), keychain=_keychain())

    assert reference.routing_trace is not None and result.routing_trace is not None
    np.testing.assert_array_equal(_get(result.outputs), _get(reference.outputs))
    np.testing.assert_array_equal(
        _get(result.routing_trace.active_expert_indices),
        _get(reference.routing_trace.active_expert_indices),
    )
    np.testing.assert_array_equal(
        _get(result.routing_trace.router_logits), _get(reference.routing_trace.router_logits)
    )
    assert result.updated_routing_state is None


# ------------------------------------------------------------------------------------------ phase mask


@pytest.mark.parametrize("mode", MOE_MODES)
@pytest.mark.usefixtures("fake_mesh")
def test_generation_mask_defaults_to_the_kernel_mode(mode: ForwardPassMode) -> None:
    # Spec: without a mask, MULTI_TOKEN tokens are context and SINGLE_TOKEN tokens are generated. A rule
    # gated on GENERATION therefore fires on a bare decode step and stays silent on a bare prefill; an
    # explicit mask overrides the default in both directions.
    module = _with(_moe(), SwitchIntervention(phases=(RoutingPhase.GENERATION,)))
    tokens = 4 if mode == ForwardPassMode.MULTI_TOKEN else 1
    inputs = _inputs(batch=2, tokens=tokens)
    state = _batched_state(module, batch=2)

    default = module(inputs, forward_pass_config=_config(mode), routing_state=state, keychain=_keychain())
    overridden = module(
        inputs,
        forward_pass_config=_config(mode),
        generation_mask=jnp.full((2, tokens), mode == ForwardPassMode.MULTI_TOKEN),
        routing_state=state,
        keychain=_keychain(),
    )

    assert default.routing_trace is not None and overridden.routing_trace is not None
    fires_by_default = mode == ForwardPassMode.SINGLE_TOKEN
    default_fixed = np.all(_get(default.routing_trace.active_expert_indices) == np.asarray(FIXED_EXPERTS))
    overridden_fixed = np.all(_get(overridden.routing_trace.active_expert_indices) == np.asarray(FIXED_EXPERTS))
    assert default_fixed == fires_by_default
    assert overridden_fixed == (not fires_by_default)


@pytest.mark.usefixtures("fake_mesh")
def test_context_only_rule_ignores_generated_tokens() -> None:
    # Spec: `phases` selects the tokens a rule may touch; a CONTEXT-only rule under a mask that marks every
    # token as generated must not fire, whatever the kernel path.
    module = _with(_moe(), SwitchIntervention(phases=(RoutingPhase.CONTEXT,)))
    inputs = _inputs(batch=2, tokens=4)
    reference = _moe()(inputs, forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN), keychain=_keychain())

    result = module(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        generation_mask=jnp.ones((2, 4), dtype=jnp.bool_),
        routing_state=_batched_state(module, batch=2),
        keychain=_keychain(),
    )

    assert result.routing_trace is not None and reference.routing_trace is not None
    np.testing.assert_array_equal(
        _get(result.routing_trace.active_expert_indices),
        _get(reference.routing_trace.active_expert_indices),
    )


# ------------------------------------------------------------------------------------------ state


@pytest.mark.usefixtures("fake_mesh")
def test_counting_state_advances_once_per_active_valid_token() -> None:
    # By construction of the mask and the lengths: row 0 has 6 valid tokens, active from position 2 -> 4;
    # row 1 has 3 valid tokens, active from position 2 -> 1. Padding must never advance the state.
    module = _with(_moe(), CountingIntervention(phases=(RoutingPhase.GENERATION,)))
    inputs = _inputs(batch=2, tokens=6)
    lengths = jnp.asarray([6, 3], dtype=jnp.int32)

    result = module(
        inputs,
        lengths_without_padding=lengths,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        generation_mask=_mask(2, 6, active_from=2),
        routing_state=_batched_state(module, batch=2),
        keychain=_keychain(),
    )

    assert isinstance(result.updated_routing_state, CountState)
    np.testing.assert_array_equal(_get(result.updated_routing_state.count), np.asarray([4, 1]))


@pytest.mark.usefixtures("fake_mesh")
def test_state_carried_across_prefill_chunks_and_decode_steps_matches_one_pass() -> None:
    # Metamorphic relation: splitting the same sequence into prefill chunks or single decode steps and
    # carrying the state must give the same final state as one pass -- the state is a function of the
    # token stream, not of how it was chunked.
    module = _with(_moe(), CountingIntervention(phases=(RoutingPhase.GENERATION,)))
    inputs = _inputs(batch=2, tokens=6)
    mask = _mask(2, 6, active_from=1)
    one_pass = module(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        generation_mask=mask,
        routing_state=_batched_state(module, batch=2),
        keychain=_keychain(),
    )

    chunked_state = _batched_state(module, batch=2)
    for start, end in ((0, 2), (2, 6)):
        chunked_state = module(
            inputs[:, start:end],
            forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
            generation_mask=mask[:, start:end],
            routing_state=chunked_state,
            keychain=_keychain(),
        ).updated_routing_state

    stepped_state = _batched_state(module, batch=2)
    for position in range(6):
        stepped_state = module(
            inputs[:, position : position + 1],
            forward_pass_config=_config(ForwardPassMode.SINGLE_TOKEN),
            generation_mask=mask[:, position : position + 1],
            routing_state=stepped_state,
            keychain=_keychain(),
        ).updated_routing_state

    assert isinstance(one_pass.updated_routing_state, CountState)
    assert isinstance(chunked_state, CountState) and isinstance(stepped_state, CountState)
    np.testing.assert_array_equal(_get(one_pass.updated_routing_state.count), np.asarray([5, 5]))
    np.testing.assert_array_equal(_get(chunked_state.count), _get(one_pass.updated_routing_state.count))
    np.testing.assert_array_equal(_get(stepped_state.count), _get(one_pass.updated_routing_state.count))


@pytest.mark.usefixtures("fake_mesh")
def test_switch_rule_fires_on_the_first_active_token_and_stays_causal() -> None:
    # Spec of the double: base routing before the first active token, FIXED_EXPERTS from it on. Causality
    # of the scan: routing up to position q must not depend on tokens after q.
    module = _with(_moe(), SwitchIntervention(phases=(RoutingPhase.GENERATION,)))
    inputs = _inputs(batch=2, tokens=6)
    reference = _moe()(inputs, forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN), keychain=_keychain())
    active_from = 3

    result = module(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        generation_mask=_mask(2, 6, active_from=active_from),
        routing_state=_batched_state(module, batch=2),
        keychain=_keychain(),
    )
    assert result.routing_trace is not None and reference.routing_trace is not None
    indices = _get(result.routing_trace.active_expert_indices)
    np.testing.assert_array_equal(
        indices[:, :active_from], _get(reference.routing_trace.active_expert_indices)[:, :active_from]
    )
    np.testing.assert_array_equal(
        indices[:, active_from:], np.broadcast_to(np.asarray(FIXED_EXPERTS), (2, 6 - active_from, NUM_ACTIVE))
    )

    altered_inputs = inputs.at[:, active_from + 1 :].set(_inputs(batch=2, tokens=6, seed=99)[:, active_from + 1 :])
    altered = module(
        altered_inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        generation_mask=_mask(2, 6, active_from=active_from),
        routing_state=_batched_state(module, batch=2),
        keychain=_keychain(),
    )
    assert altered.routing_trace is not None
    np.testing.assert_array_equal(
        _get(altered.routing_trace.active_expert_indices)[:, : active_from + 1],
        indices[:, : active_from + 1],
    )


@pytest.mark.parametrize("replicated_experts", [False, True], ids=["chunked-dispatch", "ragged-dispatch"])
@pytest.mark.usefixtures("fake_mesh")
def test_prefill_and_decode_route_a_stateful_rule_identically(replicated_experts: bool) -> None:
    # The claim the whole design rests on: a causal rule inside the parallel prefill routes every token
    # exactly as the token-by-token decode does, with the same final state.
    module = _with(_moe(replicated_experts=replicated_experts), SwitchIntervention(phases=(RoutingPhase.GENERATION,)))
    inputs = _inputs(batch=2, tokens=6)
    mask = jnp.asarray([[False, False, True, False, False, False], [False, False, False, False, True, False]])

    prefill = module(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        generation_mask=mask,
        routing_state=_batched_state(module, batch=2),
        keychain=_keychain(),
    )

    state = _batched_state(module, batch=2)
    stepped_indices = []
    for position in range(6):
        step = module(
            inputs[:, position : position + 1],
            forward_pass_config=_config(ForwardPassMode.SINGLE_TOKEN),
            generation_mask=mask[:, position : position + 1],
            routing_state=state,
            keychain=_keychain(),
        )
        assert step.routing_trace is not None
        stepped_indices.append(_get(step.routing_trace.active_expert_indices))
        state = step.updated_routing_state

    assert prefill.routing_trace is not None
    np.testing.assert_array_equal(
        _get(prefill.routing_trace.active_expert_indices), np.concatenate(stepped_indices, axis=1)
    )
    assert isinstance(prefill.updated_routing_state, SeenState) and isinstance(state, SeenState)
    np.testing.assert_array_equal(_get(prefill.updated_routing_state.seen), _get(state.seen))
    np.testing.assert_array_equal(_get(state.seen), np.asarray([True, True]))


@pytest.mark.parametrize("mode", MOE_MODES)
@pytest.mark.parametrize("replicated_experts", [False, True], ids=["chunked-dispatch", "ragged-dispatch"])
@pytest.mark.usefixtures("fake_mesh")
def test_dispatch_uses_the_intervened_routing_the_trace_reports(
    mode: ForwardPassMode, replicated_experts: bool
) -> None:
    # Differential check with an independent float64 expert reference: the output must be the weighted sum
    # over exactly the experts and weights the trace names -- here the ones the rule forced, not the router's.
    module = _with(
        _moe(replicated_experts=replicated_experts),
        SwitchIntervention(phases=(RoutingPhase.CONTEXT, RoutingPhase.GENERATION)),
    )
    tokens = 4 if mode == ForwardPassMode.MULTI_TOKEN else 1
    inputs = _inputs(batch=2, tokens=tokens)

    result = module(
        inputs,
        forward_pass_config=_config(mode),
        routing_state=_batched_state(module, batch=2),
        keychain=_keychain(),
    )
    assert result.routing_trace is not None
    indices = _get(result.routing_trace.active_expert_indices)
    assert np.all(indices == np.asarray(FIXED_EXPERTS))

    up = np.asarray(jax.device_get(module.routed_experts.up_projection.weights.decompress()), dtype=np.float64)
    down = np.asarray(jax.device_get(module.routed_experts.down_projection.weights.decompress()), dtype=np.float64)
    x = _get(inputs).astype(np.float64)
    reference = np.zeros_like(x)
    for b in range(2):
        for t in range(tokens):
            for expert in FIXED_EXPERTS:
                projected = up[expert] @ x[b, t]
                gate_half, up_half = projected[:HIDDEN_DIM], projected[HIDDEN_DIM:]
                reference[b, t] += (down[expert] @ (gate_half * up_half)) / NUM_ACTIVE
    np.testing.assert_allclose(_get(result.outputs), reference, atol=1e-4)


@pytest.mark.usefixtures("fake_mesh")
def test_stateful_rule_refuses_to_run_without_its_state() -> None:
    # Running a stateful rule from an implicit fresh state on every call would silently make decode
    # memoryless; the module demands the state instead.
    module = _with(_moe(), CountingIntervention(phases=(RoutingPhase.GENERATION,)))
    with pytest.raises(ValueError, match="routing state"):
        module(
            _inputs(batch=2, tokens=1), forward_pass_config=_config(ForwardPassMode.SINGLE_TOKEN), keychain=_keychain()
        )


# ------------------------------------------------------------------------------------------ plumbing


@pytest.mark.usefixtures("fake_mesh")
def test_with_routing_intervention_rewrites_every_moe_config_and_keeps_the_weights() -> None:
    # Functional overlay: the routing intervention lands in every mixture's config, arrays are untouched,
    # and None takes it away again.
    base = _moe()
    tree = (base, {"nested": base})
    intervention = IdentityRoutingIntervention(phases=(RoutingPhase.GENERATION,))

    overlaid = with_routing_intervention(tree, intervention)
    restored = with_routing_intervention(overlaid, None)

    for module in (overlaid[0], overlaid[1]["nested"]):
        assert module.config.routing_intervention == intervention
    assert restored[0].config.routing_intervention is None
    base_leaves = jax.tree.leaves(base)
    for module in (overlaid[0], restored[0]):
        for before, after in zip(base_leaves, jax.tree.leaves(module), strict=True):
            np.testing.assert_array_equal(_get(before), _get(after))


def test_intervention_round_trips_through_json_with_a_stable_digest() -> None:
    # The spec files the tools pass around are the JSON of the intervention; the digest is its identity in
    # run metadata and caches, so equal specs must digest equally and different phases differently.
    spec = IdentityRoutingIntervention(phases=(RoutingPhase.GENERATION,))
    restored = RoutingIntervention.from_json(spec.to_json())
    assert restored == spec
    assert RoutingIntervention.from_json({"type": "IdentityRoutingIntervention", "phases": ["context"]}) == (
        IdentityRoutingIntervention(phases=(RoutingPhase.CONTEXT,))
    )
    assert spec.digest() == IdentityRoutingIntervention(phases=(RoutingPhase.GENERATION,)).digest()
    assert spec.digest() != IdentityRoutingIntervention(phases=()).digest()


@pytest.mark.usefixtures("fake_mesh")
def test_transformer_layer_state_is_a_composite_of_mixer_and_routing_state() -> None:
    # The per-layer state gains a routing slot next to the mixer's; the batch axis is added to both.
    layer_state = TransformerLayerState(mixer=CountState(count=jnp.zeros((3,), jnp.int32)), routing=None)
    assert layer_state.routing is None
    leaves = jax.tree.leaves(layer_state)
    assert len(leaves) == 1 and leaves[0].shape == (3,)
