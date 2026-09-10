"""Tests for how the chunked MoE prefill path sums the k expert contributions of a token.

The sum used to be `.at[token_indices].add(...)` inside the scan over chunks -- a scatter whose indices
collide k times per token, which XLA runs with atomics, so the result depended on block scheduling. It
is now a gather: `flatnonzero` packs the tokens an expert was given into the leading slots, so a
contribution's position inside its expert's row is the prefix sum of the assignment mask, and from that
position the chunk and the offset within it follow. The gathered contributions are then summed over the
slot axis.

That derivation is what these tests attack. The position arithmetic is reconstructed by hand from
`token_mask`, so an off-by-one, a wrong chunk boundary or a padded slot leaking into a live token would
all produce valid-looking numbers rather than an error. Reproducibility itself is a GPU property and is
measured separately; here the target is the indexing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.mlp import MLPForwardPassConfig
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.modules.test_moe_routing_trace import HIDDEN_DIM, MODEL_DIM, _moe, _trace_inputs

# Sharded experts are the condition for the chunked dispatch: `use_ragged` also demands unsharded
# weights, so the default `_moe()` is exactly the path under test.
CHUNKED_MOE = {"replicated_experts": False}


def _run(module, inputs, *, ratio: float, lengths=None, mode=ForwardPassMode.MULTI_TOKEN):  # noqa: ANN001, ANN202
    return module(
        inputs,
        lengths_without_padding=lengths,
        forward_pass_config=MLPForwardPassConfig(mode=mode, moe_chunk_size_ratio=ratio),
        keychain=Keychain.init(7, sharding_config=make_test_sharding_config()),
    ).outputs


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize("ratio", [0.5, 0.3, 0.25], ids=["two-chunks", "three-chunks-uneven", "four-chunks"])
def test_the_result_does_not_depend_on_the_chunk_count(ratio: float) -> None:
    # The chunk count is what the new indexing is built on: position // chunk_size selects the chunk and
    # position % chunk_size the slot inside it. Splitting the same work differently must not change the
    # answer, so this is the direct check on that arithmetic. `ratio=0.3` is deliberate: 8 flattened
    # tokens give chunk_size 3, which does not divide them, so the padded tail of the last chunk is
    # exercised too.
    module = _moe(**CHUNKED_MOE)
    inputs = _trace_inputs(batch=2, tokens=4)

    single_chunk = np.asarray(jax.device_get(_run(module, inputs, ratio=1.0)), dtype=np.float32)
    split = np.asarray(jax.device_get(_run(module, inputs, ratio=ratio)), dtype=np.float32)

    # bfloat16 accumulation, one mantissa step is 2**-8; the chunk order changes nothing else.
    np.testing.assert_allclose(split, single_chunk, rtol=2.0**-8, atol=2.0**-8)


@pytest.mark.usefixtures("fake_mesh")
def test_the_padded_tail_contributes_nothing_to_the_real_tokens() -> None:
    # A padded token is assigned to no expert, so its prefix-sum position is meaningless and the mask has
    # to drop it. If it did not, the gather would still return a number -- some other token's
    # contribution -- and shapes would stay valid, which is precisely the failure mode worth a test.
    module = _moe(**CHUNKED_MOE)
    tokens = 4
    inputs = _trace_inputs(batch=2, tokens=tokens)
    full = jax.device_put(jnp.full((2,), tokens, dtype=jnp.int32), make_sharding((LogicalAxis.BATCH,)))
    truncated = jax.device_put(jnp.asarray([2, tokens], dtype=jnp.int32), make_sharding((LogicalAxis.BATCH,)))

    with_full = np.asarray(jax.device_get(_run(module, inputs, ratio=0.5, lengths=full)), dtype=np.float32)
    with_padding = np.asarray(jax.device_get(_run(module, inputs, ratio=0.5, lengths=truncated)), dtype=np.float32)

    # Row 1 is unpadded in both calls; row 0 keeps its first two tokens live in both.
    np.testing.assert_allclose(with_padding[1], with_full[1], rtol=2.0**-8, atol=2.0**-8)
    np.testing.assert_allclose(with_padding[0, :2], with_full[0, :2], rtol=2.0**-8, atol=2.0**-8)


@pytest.mark.usefixtures("fake_mesh")
def test_the_chunked_path_agrees_with_the_decode_path() -> None:
    # Decode sums the k contributions with a reduction over a fixed axis and never used a scatter, so it
    # is the reference the rewritten chunked sum has to reproduce.
    module = _moe(**CHUNKED_MOE)
    inputs = _trace_inputs(batch=2, tokens=1)

    prefill = _run(module, inputs, ratio=0.5)
    decode = _run(module, inputs, ratio=0.5, mode=ForwardPassMode.SINGLE_TOKEN)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(prefill), dtype=np.float32),
        np.asarray(jax.device_get(decode), dtype=np.float32),
        rtol=2.0**-8,
        atol=2.0**-8,
    )


@pytest.mark.usefixtures("fake_mesh")
def test_a_token_gets_its_own_k_contributions_when_the_chunks_split_its_experts() -> None:
    # Independent float64 reference for one token, computed with several chunks in play so that a token's
    # experts land in different chunks. The module output must equal the softmax-weighted sum over the
    # traced experts: a gather that picked the right expert but the wrong offset, or the right offset in
    # the wrong chunk, would show up here as a mismatch instead of as plausible numbers.
    module = _moe(**CHUNKED_MOE)
    inputs = _trace_inputs(batch=2, tokens=4)

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.MULTI_TOKEN, moe_chunk_size_ratio=0.25),
        keychain=Keychain.init(7, sharding_config=make_test_sharding_config()),
    )
    trace = result.routing_trace
    assert trace is not None

    up = np.asarray(jax.device_get(module.routed_experts.up_projection.weights.decompress()), dtype=np.float64)
    down = np.asarray(jax.device_get(module.routed_experts.down_projection.weights.decompress()), dtype=np.float64)
    x = np.asarray(jax.device_get(inputs), dtype=np.float64)
    logits = np.asarray(jax.device_get(trace.router_logits), dtype=np.float64)
    indices = np.asarray(jax.device_get(trace.active_expert_indices))
    outputs = np.asarray(jax.device_get(result.outputs), dtype=np.float64)

    for row, token in ((0, 0), (1, 3)):
        active = indices[row, token]
        active_logits = logits[row, token, active]
        weights = np.exp(active_logits - active_logits.max())
        weights /= weights.sum()
        expected = np.zeros(MODEL_DIM, dtype=np.float64)
        for weight, expert in zip(weights, active, strict=True):
            # Identity activation, no biases: expert(x) = down @ (gate_half * up_half), and the up
            # projection stacks [gate; up] halves.
            projected = up[expert] @ x[row, token]
            gate_half, up_half = projected[:HIDDEN_DIM], projected[HIDDEN_DIM:]
            expected += weight * (down[expert] @ (gate_half * up_half))
        np.testing.assert_allclose(outputs[row, token], expected, rtol=2.0**-6, atol=2.0**-6)


@pytest.mark.usefixtures("fake_mesh")
def test_a_padded_token_gets_exactly_zero() -> None:
    # The old scatter dropped padded slots by their sentinel index, so a padded row came out as exact
    # zero. The gather has no such fallback: a padded token's prefix-sum position points at whichever
    # token was assigned last before it, so without the mask it would receive that token's contribution.
    # Nothing downstream re-masks the MoE output, so this row would carry junk into the residual stream.
    # Exact zero, not a tolerance: the mask either applies or it does not.
    module = _moe(**CHUNKED_MOE)
    tokens = 4
    inputs = _trace_inputs(batch=2, tokens=tokens)
    lengths = jax.device_put(jnp.asarray([2, tokens], dtype=jnp.int32), make_sharding((LogicalAxis.BATCH,)))

    outputs = np.asarray(jax.device_get(_run(module, inputs, ratio=0.5, lengths=lengths)), dtype=np.float32)

    np.testing.assert_array_equal(outputs[0, 2:], np.zeros((tokens - 2, MODEL_DIM), dtype=np.float32))
