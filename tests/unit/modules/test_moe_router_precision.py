"""Tests for the precision of the MoE router.

Expectations come from the arithmetic, not from running the code. The router is computed in float32
regardless of the activation dtype, so that the top-k sees gaps a bfloat16 logit cannot represent.
The dtype that leaves the layer must stay the activation dtype, and that is asserted separately,
because "it also changed X" is the failure mode that would silently widen a measurement.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.mlp import MixtureOfExperts, MLPResult
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.modules.test_moe_routing_trace import (
    MODEL_DIM,
    NUM_ACTIVE,
    NUM_ROUTED_EXPERTS,
    _linear,
    _moe,
    _trace_inputs,
)


def _call(module: MixtureOfExperts, inputs: jax.Array) -> MLPResult:
    return module.call_with_routing(inputs, keychain=Keychain.init(7, sharding_config=make_test_sharding_config()))


@pytest.mark.usefixtures("fake_mesh")
def test_router_logits_are_float32_and_the_layer_keeps_its_own_dtype() -> None:
    # `FullPrecisionMatrix.dot` casts the weights to the dtype of the input, so widening the input is
    # what moves the matmul -- and therefore the logits -- to float32. The layer's own output must stay
    # bfloat16: a float32 result would promote the residual stream behind it.
    inputs = _trace_inputs(batch=2, tokens=5).astype(jnp.bfloat16)

    result = _call(_moe(), inputs)

    assert result.routing_trace is not None
    assert result.routing_trace.router_logits.dtype == jnp.float32
    assert result.outputs.dtype == jnp.bfloat16


@pytest.mark.usefixtures("fake_mesh")
def test_a_gap_below_one_bfloat16_step_decides_the_top_k() -> None:
    # Constructed so the arithmetic decides the answer. The input is [1, 1, 0, 0] and the biases are
    # zero, so the logit of expert e is W[e, 0] + W[e, 1]. Experts 0 and 1 get 1.5 and 1.25, exact in
    # bfloat16, and they take two of the NUM_ACTIVE = 3 slots under either precision. The last slot is
    # decided by experts 4 and 5, at 1 + 2^-10 and 1 + 2^-9: bfloat16 keeps 7 stored mantissa bits, so
    # the step above 1.0 is 2^-7 and everything below 1 + 2^-8 rounds back to 1.0. A bfloat16 router
    # would see both -- and the four untouched experts -- as exactly 1.0, and the tie would fall to the
    # lowest index, expert 2. In float32 the two are distinct and expert 5 is the largest of them, so it
    # takes the slot.
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

    result = _call(module, inputs)

    assert result.routing_trace is not None
    selected = set(np.asarray(jax.device_get(result.routing_trace.active_expert_indices)).ravel().tolist())
    assert selected == {0, 1, 5}
    assert len(selected) == NUM_ACTIVE
