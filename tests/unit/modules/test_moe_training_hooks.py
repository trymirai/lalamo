"""Tests for the hooks a training loop needs from the MoE routing path.

Four things are under test: the routing scan of a softmax-weighted rule runs on detached logits with the
weights recomputed outside it; the trace carries the logits the selection was taken over (`effective_logits`)
and the activations the router read (`router_inputs`); `Decoder.features` is `__call__` without the readout;
the rematerialisation flags change nothing in the forward pass.

Expectations are derived from the contracts and from independent references, never from the code under test:
the paper's LRU rule in plain Python for the selection and the boost, a float64 expert reference for the
dispatch, a JAX reference with the selection held constant for the gradient, and bit-exact equality wherever
the change is meant to be invisible.
"""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.decoder import DecoderForwardPassConfig
from lalamo.modules.mlp import ROUTING_RESIDUAL, MLPForwardPassConfig, RoutingPhase
from lalamo.modules.routing_interventions import CacheConditionalRouting, ExpertCacheState
from lalamo.modules.transformer_layer import TransformerForwardPassConfig
from lalamo.modules.utils import call_vmapped_twice
from tests.helpers import build_tiny_attention_decoder
from tests.unit.modules.test_cache_conditional_routing import BOTH_PHASES, ReferenceCache, _cache_of
from tests.unit.modules.test_moe_routing_intervention import (
    _batched_state,
    _config,
    _get,
    _inputs,
    _keychain,
    _mask,
    _with,
)
from tests.unit.modules.test_moe_routing_trace import HIDDEN_DIM, NUM_ACTIVE, _moe

RULE = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.7, forced_top=1, cache_capacity=3)


def _reference_walk(
    logits: np.ndarray,
    active: np.ndarray,
    rule: CacheConditionalRouting,
) -> tuple[np.ndarray, np.ndarray, list[ReferenceCache]]:
    """The paper's rule step by step: dispatched indices, the boosted logits it selected over, final caches."""
    tokens, batch, num_experts = logits.shape
    references = [ReferenceCache() for _ in range(batch)]
    indices = np.zeros((batch, tokens, NUM_ACTIVE), dtype=np.int64)
    effective = np.zeros((batch, tokens, num_experts), dtype=np.float32)
    capacity = rule.capacity(num_experts)
    for step in range(tokens):
        for row in range(batch):
            reference = references[row]
            logits32 = logits[step, row].astype(np.float32)
            token_range = np.float32(logits32.max() - logits32.min())
            delta = np.float32(np.mean(reference.ranges)) if reference.ranges else token_range
            ranking = sorted(range(num_experts), key=lambda expert: -logits32[expert])
            boosted_set = reference.cached(capacity) | set(ranking[: rule.forced_top])
            mask = np.asarray([expert in boosted_set for expert in range(num_experts)], dtype=np.float32)
            boosted = logits32 + np.float32(rule.bias) * delta * mask
            effective[row, step] = boosted if active[step, row] else logits32
            selected, _ = reference.step(
                logits[step, row],
                bool(active[step, row]),
                bias=rule.bias,
                forced_top=rule.forced_top,
                capacity=capacity,
                num_active=NUM_ACTIVE,
                logit_range=rule.logit_range,
            )
            indices[row, step] = selected
    return indices, effective, references


@pytest.mark.usefixtures("fake_mesh")
def test_detached_scan_selects_like_the_lru_reference_and_dispatches_softmax_weights() -> None:
    # Contract of the change: the selection, the eviction order and the final cache are those of the paper's
    # rule (reference in plain Python), and the dispatch weighs the selected experts by the softmax of the
    # router logits over them (float64 expert reference) -- the weights the scan no longer returns.
    module = _with(_moe(), RULE)  # active on both phases: every token is boosted
    batch, tokens = 2, 10
    inputs = _inputs(batch=batch, tokens=tokens, seed=21)
    result = module(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        routing_state=_batched_state(module, batch),
        keychain=_keychain(),
    )
    trace = result.routing_trace
    assert trace is not None
    logits = _get(trace.router_logits)  # [batch, tokens, experts], float32
    every_token = np.ones((tokens, batch), dtype=np.bool_)
    indices, _, references = _reference_walk(np.transpose(logits, (1, 0, 2)), every_token, RULE)
    np.testing.assert_array_equal(_get(trace.active_expert_indices), indices)
    assert isinstance(result.updated_routing_state, ExpertCacheState)
    for row in range(batch):
        assert _cache_of(result.updated_routing_state, row, RULE.capacity(logits.shape[-1])) == references[row].cached(
            RULE.capacity(logits.shape[-1])
        )

    up = np.asarray(jax.device_get(module.routed_experts.up_projection.weights.decompress()), dtype=np.float64)
    down = np.asarray(jax.device_get(module.routed_experts.down_projection.weights.decompress()), dtype=np.float64)
    x = _get(inputs).astype(np.float64)
    reference = np.zeros_like(x)
    for row in range(batch):
        for step in range(tokens):
            chosen = logits[row, step, indices[row, step]].astype(np.float64)
            weights = np.exp(chosen - chosen.max())
            weights /= weights.sum()
            for weight, expert in zip(weights, indices[row, step], strict=True):
                projected = up[expert] @ x[row, step]
                reference[row, step] += weight * (down[expert] @ (projected[:HIDDEN_DIM] * projected[HIDDEN_DIM:]))
    # fp32 forward against a float64 reference on 4-dimensional vectors: 1e-4 is generous.
    np.testing.assert_allclose(_get(result.outputs), reference, atol=1e-4)


@pytest.mark.usefixtures("fake_mesh")
def test_effective_logits_are_the_boosted_field_the_selection_was_taken_over() -> None:
    # Contract of the trace field: top_k(effective_logits) is the dispatched set; on inactive tokens it is the
    # plain logits; on active ones it is logits + bias * delta over the cached-or-forced experts of the paper's
    # rule, as reconstructed by the Python reference from its own cache and running range.
    module = _with(_moe(), replace(RULE, phases=(RoutingPhase.GENERATION,)))
    batch, tokens = 2, 9
    inputs = _inputs(batch=batch, tokens=tokens, seed=22)
    mask = _mask(batch, tokens, active_from=4)
    trace = module(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        generation_mask=mask,
        routing_state=_batched_state(module, batch),
        keychain=_keychain(),
    ).routing_trace
    assert trace is not None
    logits = _get(trace.router_logits)
    effective = _get(trace.effective_logits)
    indices = _get(trace.active_expert_indices)

    _, expected_effective, _ = _reference_walk(
        np.transpose(logits, (1, 0, 2)),
        np.transpose(_get(mask), (1, 0)),
        replace(RULE, phases=(RoutingPhase.GENERATION,)),
    )
    # fp32 arithmetic on both sides in a different order: mantissa 1.2e-7 on values of order 1.
    np.testing.assert_allclose(effective, expected_effective, atol=1e-5)
    np.testing.assert_array_equal(effective[:, :4], logits[:, :4])
    assert np.any(effective[:, 4:] != logits[:, 4:])  # the boost really fired somewhere
    for row in range(batch):
        for step in range(tokens):
            top = set(np.argsort(-effective[row, step])[:NUM_ACTIVE].tolist())
            assert top == set(indices[row, step].tolist()), (row, step)


@pytest.mark.usefixtures("fake_mesh")
def test_router_inputs_in_the_trace_reproduce_the_router_logits() -> None:
    # Contract: applying the module's own router to the traced inputs gives the traced logits, so a second
    # router (a frozen reference) applied to the same inputs is comparable with them.
    module = _with(_moe(), RULE)
    inputs = _inputs(batch=2, tokens=5, seed=23)
    trace = module(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        routing_state=_batched_state(module, 2),
        keychain=_keychain(),
    ).routing_trace
    assert trace is not None
    weights = np.asarray(jax.device_get(module.router.weights.decompress()), dtype=np.float64)
    biases = np.asarray(jax.device_get(module.router.biases), dtype=np.float64)
    reconstructed = _get(trace.router_inputs).astype(np.float64) @ weights.T + biases
    np.testing.assert_allclose(_get(trace.router_logits), reconstructed, atol=1e-5)
    np.testing.assert_array_equal(_get(trace.router_inputs), _get(inputs))


@pytest.mark.parametrize("replicated_experts", [False, True], ids=["chunked-dispatch", "ragged-dispatch"])
@pytest.mark.usefixtures("fake_mesh")
def test_router_gradient_equals_the_fixed_selection_reference_and_needs_no_reverse_scan(
    replicated_experts: bool,
) -> None:
    # The gradient of the MoE output w.r.t. the router weights must equal that of an independent JAX reference
    # in which the dispatched set is a constant and the weights are the softmax of the gathered logits -- the
    # only differentiable path a top-k router has. And because the scan runs on detached logits, reverse mode
    # must contain no transposed (reverse=True) scan at all.
    module = _with(_moe(replicated_experts=replicated_experts), RULE)
    batch, tokens = 2, 6
    inputs = _inputs(batch=batch, tokens=tokens, seed=24)
    probe = jax.device_put(jax.random.normal(jax.random.PRNGKey(3), inputs.shape, dtype=jnp.float32), inputs.sharding)
    router_weights = module.router.weights.weights
    routing_state = _batched_state(module, batch)

    def module_loss(weights: Array) -> Array:
        rebuilt = eqx.tree_at(lambda m: m.router.weights.weights, module, weights)
        outputs = rebuilt(
            inputs,
            forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
            routing_state=routing_state,
            keychain=_keychain(),
        ).outputs
        return (outputs * probe).sum()

    trace = module(
        inputs,
        forward_pass_config=_config(ForwardPassMode.MULTI_TOKEN),
        routing_state=routing_state,
        keychain=_keychain(),
    ).routing_trace
    assert trace is not None
    fixed_indices = jnp.asarray(_get(trace.active_expert_indices))
    up = jnp.asarray(jax.device_get(module.routed_experts.up_projection.weights.decompress()))
    down = jnp.asarray(jax.device_get(module.routed_experts.down_projection.weights.decompress()))
    biases = jnp.asarray(jax.device_get(module.router.biases))
    x = jnp.asarray(_get(inputs))
    probe_host = jnp.asarray(_get(probe))

    def reference_loss(weights: Array) -> Array:
        logits = jnp.einsum("btd,ed->bte", x, weights) + biases
        mixing = jax.nn.softmax(jnp.take_along_axis(logits, fixed_indices, axis=-1), axis=-1)
        projected = jnp.einsum("btkhd,btd->btkh", up[fixed_indices], x)
        hidden = projected[..., :HIDDEN_DIM] * projected[..., HIDDEN_DIM:]
        expert_outputs = jnp.einsum("btkdh,btkh->btkd", down[fixed_indices], hidden)
        return ((mixing[..., None] * expert_outputs).sum(axis=2) * probe_host).sum()

    module_gradient = _get(jax.grad(module_loss)(router_weights))
    reference_gradient = _get(jax.grad(reference_loss)(jnp.asarray(_get(router_weights))))
    assert np.all(np.isfinite(module_gradient))
    # fp32 on both sides, different summation orders (chunked dispatch vs einsum): 1e-4 relative is lax.
    np.testing.assert_allclose(module_gradient, reference_gradient, rtol=1e-4, atol=1e-6)

    # The chunked dispatch has a scan of its own (over expert chunks) that reverse mode legitimately
    # transposes, so the property is relative: the routing rule adds no transposed scan to the module's
    # backward pass beyond what the plain module already has.
    plain_module = _moe(replicated_experts=replicated_experts)

    def plain_loss(weights: Array) -> Array:
        rebuilt = eqx.tree_at(lambda m: m.router.weights.weights, plain_module, weights)
        config = _config(ForwardPassMode.MULTI_TOKEN)
        return (rebuilt(inputs, forward_pass_config=config, keychain=_keychain()).outputs * probe).sum()

    with_rule = str(jax.make_jaxpr(jax.grad(module_loss))(router_weights))
    without_rule = str(jax.make_jaxpr(jax.grad(plain_loss))(router_weights))
    assert "scan" in with_rule  # the forward routing scan is there ...
    assert with_rule.count("reverse=True") == without_rule.count("reverse=True")  # ... and nothing transposes it


def test_decoder_features_are_the_call_without_the_readout() -> None:
    # Contract of the split: `__call__` == readout(features), bit for bit, including the key the readout draws.
    decoder = build_tiny_attention_decoder((None, None))
    token_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 6), 0, decoder.vocab_size, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32)[None, :], (2, 6))
    keychain = Keychain.init(5, sharding_config=decoder.sharding_config)
    config = DecoderForwardPassConfig()

    result = decoder(token_ids, positions, forward_pass_config=config, keychain=keychain)
    features = decoder.features(token_ids, positions, forward_pass_config=config, keychain=keychain)
    logits = call_vmapped_twice(
        decoder.embedding.readout,
        features.transformer_result.outputs,
        forward_pass_config=config.embedding_forward_pass_config,
        keychain=features.readout_keychain,
        added_sharding_axes=(decoder.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
    )
    np.testing.assert_array_equal(_get(logits), _get(result.logits))
    assert features.transformer_result.outputs.shape == (2, 6, decoder.transformer.config.model_dim)


def test_layer_remat_leaves_the_decoder_forward_bit_exact() -> None:
    # A rematerialisation flag decides what is kept for the backward pass; the forward values must not move.
    decoder = build_tiny_attention_decoder((None, None))
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (2, 6), 0, decoder.vocab_size, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32)[None, :], (2, 6))
    keychain = Keychain.init(5, sharding_config=decoder.sharding_config)
    plain = DecoderForwardPassConfig()
    remat = replace(plain, transformer_forward_pass_config=replace(TransformerForwardPassConfig(), remat_layers=True))

    np.testing.assert_array_equal(
        _get(decoder(token_ids, positions, forward_pass_config=remat, keychain=keychain).logits),
        _get(decoder(token_ids, positions, forward_pass_config=plain, keychain=keychain).logits),
    )


@pytest.mark.usefixtures("fake_mesh")
def test_expert_chunk_remat_leaves_the_moe_forward_bit_exact() -> None:
    # Same contract on the chunked dispatch (the path every quantised checkpoint takes in prefill).
    module = _with(_moe(), RULE)
    inputs = _inputs(batch=2, tokens=8, seed=25)
    plain = MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.5)
    remat = replace(plain, remat_expert_chunks=True)
    outputs = [
        module(
            inputs, forward_pass_config=config, routing_state=_batched_state(module, 2), keychain=_keychain()
        ).outputs
        for config in (plain, remat)
    ]
    np.testing.assert_array_equal(_get(outputs[0]), _get(outputs[1]))


@pytest.mark.parametrize("unroll", [4, 8])
@pytest.mark.usefixtures("fake_mesh")
def test_unrolling_the_routing_scan_changes_no_bit(unroll: int) -> None:
    # `unroll` only regroups the scan's iterations into fewer loop trips; the operations and their order are
    # the same, so outputs, dispatched experts and the final cache must be identical -- including a length
    # the unroll factor does not divide.
    module = _with(_moe(), RULE)
    inputs = _inputs(batch=2, tokens=10, seed=26)
    plain = MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.5)
    results = [
        module(inputs, forward_pass_config=config, routing_state=_batched_state(module, 2), keychain=_keychain())
        for config in (plain, replace(plain, routing_scan_unroll=unroll))
    ]
    reference, unrolled = results
    assert reference.routing_trace is not None and unrolled.routing_trace is not None
    np.testing.assert_array_equal(_get(unrolled.outputs), _get(reference.outputs))
    np.testing.assert_array_equal(
        _get(unrolled.routing_trace.active_expert_indices), _get(reference.routing_trace.active_expert_indices)
    )
    assert isinstance(reference.updated_routing_state, ExpertCacheState)
    assert isinstance(unrolled.updated_routing_state, ExpertCacheState)
    np.testing.assert_array_equal(
        _get(unrolled.updated_routing_state.last_used), _get(reference.updated_routing_state.last_used)
    )


@pytest.mark.usefixtures("fake_mesh")
def test_saving_the_scan_outputs_spares_the_backward_a_second_scan() -> None:
    # Under plain rematerialisation the backward pass re-runs the layer's forward, routing scan included, so
    # the scan appears twice in the gradient program. With the policy that saves the scan's named outputs it
    # appears once -- and the gradient is the same to the bit, because saving a value never changes it.
    module = _with(_moe(), RULE)
    tokens = 7  # the expert-chunk scan of this module has length 2 (14 flattened tokens at ratio 0.5), not 7
    inputs = _inputs(batch=2, tokens=tokens, seed=27)
    probe = jax.device_put(jax.random.normal(jax.random.PRNGKey(4), inputs.shape, dtype=jnp.float32), inputs.sharding)
    router_weights = module.router.weights.weights
    routing_state = _batched_state(module, 2)
    config = _config(ForwardPassMode.MULTI_TOKEN)

    def loss_under(policy: object) -> "jax.Array":
        def loss(weights: Array) -> Array:
            rebuilt = eqx.filter_checkpoint(
                eqx.tree_at(lambda m: m.router.weights.weights, module, weights), policy=policy
            )
            outputs = rebuilt(
                inputs, forward_pass_config=config, routing_state=routing_state, keychain=_keychain()
            ).outputs
            return (outputs * probe).sum()

        return loss

    plain, saving = (
        loss_under(None),
        loss_under(jax.checkpoint_policies.save_only_these_names(ROUTING_RESIDUAL)),
    )
    plain_program = str(jax.make_jaxpr(jax.grad(plain))(router_weights))
    saving_program = str(jax.make_jaxpr(jax.grad(saving))(router_weights))
    assert plain_program.count(f"length={tokens}") == 2
    assert saving_program.count(f"length={tokens}") == 1
    np.testing.assert_array_equal(_get(jax.grad(saving)(router_weights)), _get(jax.grad(plain)(router_weights)))
