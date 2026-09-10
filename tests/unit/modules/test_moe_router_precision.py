"""Tests for `MixtureOfExpertsConfig.router_in_fp32`.

Expectations come from the arithmetic, not from running the code. The flag is supposed to do exactly
one thing: compute the router in float32 so that the top-k sees gaps a bfloat16 logit cannot represent.
Everything else -- which dtype leaves the layer, what happens when the activations are already float32,
what a checkpoint without the key deserializes to -- must be unchanged, and each of those is asserted
separately, because "it also changed X" is the failure mode that would silently widen the measurement.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.mlp import MixtureOfExperts, MixtureOfExpertsConfig
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.modules.test_moe_routing_trace import (
    MODEL_DIM,
    NUM_ACTIVE,
    NUM_ROUTED_EXPERTS,
    _linear,
    _moe,
    _trace_inputs,
)


def _with_flag(module: MixtureOfExperts, *, enabled: bool) -> MixtureOfExperts:
    return replace(module, config=replace(module.config, router_in_fp32=enabled))


def _call(module: MixtureOfExperts, inputs: jax.Array):  # noqa: ANN202
    return module(inputs, keychain=Keychain.init(7, sharding_config=make_test_sharding_config()))


@pytest.mark.usefixtures("fake_mesh")
def test_config_defaults_to_off_and_survives_a_roundtrip_without_the_key() -> None:
    # A checkpoint written before the field existed must still deserialize: the key is optional.
    config = _moe().config
    assert config.router_in_fp32 is False
    payload = config.to_json()
    assert isinstance(payload, dict)
    assert payload["router_in_fp32"] is False
    del payload["router_in_fp32"]
    assert MixtureOfExpertsConfig.from_json(payload).router_in_fp32 is False


@pytest.mark.usefixtures("fake_mesh")
def test_flag_is_inert_when_activations_are_already_float32() -> None:
    # The flag only widens the router input. With float32 activations that is a no-op, so both settings
    # must agree bit for bit -- the invariant that says the flag cannot change anything on its own.
    module = _moe()
    inputs = _trace_inputs(batch=2, tokens=5).astype(jnp.float32)

    off = _call(_with_flag(module, enabled=False), inputs)
    on = _call(_with_flag(module, enabled=True), inputs)

    np.testing.assert_array_equal(jax.device_get(off.outputs), jax.device_get(on.outputs))
    assert off.routing_trace is not None and on.routing_trace is not None
    np.testing.assert_array_equal(
        jax.device_get(off.routing_trace.active_expert_indices),
        jax.device_get(on.routing_trace.active_expert_indices),
    )


@pytest.mark.parametrize("enabled", [False, True], ids=["bf16-router", "fp32-router"])
@pytest.mark.usefixtures("fake_mesh")
def test_router_logits_take_the_configured_precision_and_the_layer_keeps_its_own(enabled: bool) -> None:
    # `FullPrecisionMatrix.dot` casts the weights to the dtype of the input, so widening the input is
    # what moves the matmul -- and therefore the logits -- to float32. The layer's own output must stay
    # bfloat16 either way: a float32 mixing weight would promote the expert combination as well.
    module = _with_flag(_moe(), enabled=enabled)
    inputs = _trace_inputs(batch=2, tokens=5).astype(jnp.bfloat16)

    result = _call(module, inputs)

    assert result.routing_trace is not None
    assert result.routing_trace.router_logits.dtype == (jnp.float32 if enabled else jnp.bfloat16)
    assert result.outputs.dtype == jnp.bfloat16


@pytest.mark.usefixtures("fake_mesh")
def test_a_gap_below_one_bfloat16_step_is_resolved_only_in_float32() -> None:
    # Constructed so the arithmetic decides the answer. The input is [1, 1, 0, 0] and the biases are
    # zero, so the logit of expert e is W[e, 0] + W[e, 1]. Experts 0 and 1 get 1.5 and 1.25, exact in
    # bfloat16, and they take two of the NUM_ACTIVE = 3 slots under either precision. The last slot is
    # decided by experts 4 and 5, at 1 + 2^-10 and 1 + 2^-9: bfloat16 keeps 7 stored mantissa bits, so
    # the step above 1.0 is 2^-7 and everything below 1 + 2^-8 rounds back to 1.0. In bfloat16 both --
    # and the four untouched experts -- are exactly 1.0, so the tie falls to the lowest index, expert 2.
    # In float32 the two are distinct and expert 5 is the largest of them, so it takes the slot.
    weights = np.zeros((NUM_ROUTED_EXPERTS, MODEL_DIM), dtype=np.float32)
    weights[:, 0] = 1.0
    weights[0, 1] = 0.5
    weights[1, 1] = 0.25
    weights[4, 1] = 2.0**-10
    weights[5, 1] = 2.0**-9
    router = _linear(jnp.asarray(weights), jnp.zeros((NUM_ROUTED_EXPERTS,), jnp.float32), (NUM_ROUTED_EXPERTS,))
    module = replace(_moe(), router=router)
    # Two rows because the mesh partitions the batch axis in two; both carry the same input.
    inputs = jax.device_put(
        jnp.zeros((2, 1, MODEL_DIM), dtype=jnp.bfloat16).at[..., 0].set(1).at[..., 1].set(1),
        make_sharding((LogicalAxis.BATCH, None, None)),
    )

    off = _call(_with_flag(module, enabled=False), inputs)
    on = _call(_with_flag(module, enabled=True), inputs)

    assert off.routing_trace is not None and on.routing_trace is not None
    selected_off = set(np.asarray(jax.device_get(off.routing_trace.active_expert_indices)).ravel().tolist())
    selected_on = set(np.asarray(jax.device_get(on.routing_trace.active_expert_indices)).ravel().tolist())
    assert selected_off == {0, 1, 2}
    assert selected_on == {0, 1, 5}
    assert len(selected_off) == NUM_ACTIVE
