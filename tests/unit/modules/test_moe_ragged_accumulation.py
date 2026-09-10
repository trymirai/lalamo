"""Tests for how the ragged MoE prefill path sums the k expert contributions of a token.

The sum used to be `.at[token_indices].add(...)`, a scatter whose indices collide k times per token;
XLA runs that with atomics, so the result depended on block scheduling and moved between runs of the
same compiled code. It is now an inverse permutation followed by a reduction over a fixed axis, which
is the same arithmetic without collisions. These tests pin the two properties that change could break:
the padded tail must contribute nothing, and the two prefill dispatches must still agree with each
other and with decode. Reproducibility itself is a GPU property and is measured separately.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.mlp import MLPForwardPassConfig
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.modules.test_moe_routing_trace import MODEL_DIM, _moe, _trace_inputs


def _run(module, inputs, *, lengths=None, mode=ForwardPassMode.MULTI_TOKEN):  # noqa: ANN001, ANN202
    return module(
        inputs,
        lengths_without_padding=lengths,
        forward_pass_config=MLPForwardPassConfig(mode=mode, moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(7, sharding_config=make_test_sharding_config()),
    ).outputs


@pytest.mark.usefixtures("fake_mesh")
def test_the_padded_tail_contributes_nothing_to_the_real_tokens() -> None:
    # The padded slots are routed to a sentinel expert and zeroed before the sum, so undoing the
    # permutation must drop them back onto their own rows and leave the real tokens untouched. If the
    # inverse permutation were off, padding would leak into a real token's sum -- shapes stay valid.
    module = _moe(replicated_experts=True)
    tokens = 4
    inputs = _trace_inputs(batch=2, tokens=tokens)
    full = jax.device_put(jnp.full((2,), tokens, dtype=jnp.int32), make_sharding((LogicalAxis.BATCH,)))
    truncated = jax.device_put(jnp.asarray([2, tokens], dtype=jnp.int32), make_sharding((LogicalAxis.BATCH,)))

    with_full = np.asarray(jax.device_get(_run(module, inputs, lengths=full)), dtype=np.float32)
    with_padding = np.asarray(jax.device_get(_run(module, inputs, lengths=truncated)), dtype=np.float32)

    # Row 1 is unpadded in both calls, and row 0's first two tokens are real in both.
    np.testing.assert_allclose(with_padding[1], with_full[1], rtol=2.0**-8, atol=2.0**-8)
    np.testing.assert_allclose(with_padding[0, :2], with_full[0, :2], rtol=2.0**-8, atol=2.0**-8)


@pytest.mark.usefixtures("fake_mesh")
def test_the_two_prefill_dispatches_agree_with_each_other() -> None:
    # Replicated experts take the ragged_dot path, sharded ones the chunked scatter path. They compute
    # the same function, so a change to either accumulation must keep them equal -- this is the check
    # that the permutation trick did not quietly reorder a token's contributions.
    inputs = _trace_inputs(batch=2, tokens=4)
    ragged = _run(_moe(replicated_experts=True), inputs)
    chunked = _run(_moe(replicated_experts=False), inputs)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(ragged), dtype=np.float32),
        np.asarray(jax.device_get(chunked), dtype=np.float32),
        # bfloat16 accumulation, one mantissa step is 2**-8.
        rtol=2.0**-8,
        atol=2.0**-8,
    )


@pytest.mark.usefixtures("fake_mesh")
def test_the_ragged_path_agrees_with_the_decode_path_on_a_single_token() -> None:
    # Decode sums the k contributions with a reduction over a fixed axis and never used a scatter, so
    # it is the reference the rewritten prefill sum has to reproduce.
    module = _moe(replicated_experts=True)
    inputs = _trace_inputs(batch=2, tokens=1)

    prefill = _run(module, inputs)
    decode = _run(module, inputs, mode=ForwardPassMode.SINGLE_TOKEN)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(prefill), dtype=np.float32),
        np.asarray(jax.device_get(decode), dtype=np.float32),
        rtol=2.0**-8,
        atol=2.0**-8,
    )


@pytest.mark.usefixtures("fake_mesh")
def test_a_token_receives_exactly_its_own_k_contributions() -> None:
    # Independent float64 reference for one token of one row: the module output must equal the softmax
    # weighted sum over the traced experts. Recomputed from the weights, so a permutation that mixed
    # two tokens' contributions would show up as a mismatch rather than as valid-looking numbers.
    module = _moe(replicated_experts=True)
    inputs = _trace_inputs(batch=2, tokens=4)

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(7, sharding_config=make_test_sharding_config()),
    )
    trace = result.routing_trace
    assert trace is not None

    up = np.asarray(jax.device_get(module.routed_experts.up_projection.weights.decompress()), dtype=np.float64)
    down = np.asarray(jax.device_get(module.routed_experts.down_projection.weights.decompress()), dtype=np.float64)
    x = np.asarray(jax.device_get(inputs), dtype=np.float64)
    logits = np.asarray(jax.device_get(trace.router_logits), dtype=np.float64)
    indices = np.asarray(jax.device_get(trace.active_expert_indices))
    hidden = up.shape[1] // 2

    row, token = 1, 2
    active = indices[row, token]
    active_logits = logits[row, token, active]
    weights = np.exp(active_logits - active_logits.max())
    weights /= weights.sum()
    expected = np.zeros(MODEL_DIM, dtype=np.float64)
    for weight, expert in zip(weights, active, strict=True):
        projected = up[expert] @ x[row, token]
        gate_half, up_half = projected[:hidden], projected[hidden:]
        expected += weight * (down[expert] @ (gate_half * up_half))

    np.testing.assert_allclose(
        np.asarray(jax.device_get(result.outputs), dtype=np.float64)[row, token],
        expected,
        rtol=2.0**-6,
        atol=2.0**-6,
    )
