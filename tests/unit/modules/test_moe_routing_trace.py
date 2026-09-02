"""Tests for MixtureOfExperts.routing_trace / MLPBase.routing_trace.

The trace is a shadow pass over the router, so every expectation here is derived from an
independent float64 numpy reimplementation of the router math (weights @ x + bias, then top-k by
sorting) -- it never calls the module under test, so a sign flip, a transposed axis or a dropped
bias in the traced path cannot be masked by the reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.activations import Identity
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.mlp import (
    DenseMLP,
    DenseMLPConfig,
    MixtureOfExperts,
    MixtureOfExpertsConfig,
    MLPForwardPassConfig,
    SoftmaxRouting,
)
from lalamo.weight_matrix import FullPrecisionSpec
from tests.helpers import make_sharding, make_test_sharding_config

MODEL_DIM = 4
HIDDEN_DIM = 4
NUM_ROUTED_EXPERTS = 8
NUM_ACTIVE = 3

MOE_MODES = [
    pytest.param(ForwardPassMode.MULTI_TOKEN, id="multi-token"),
    pytest.param(ForwardPassMode.SINGLE_TOKEN, id="single-token"),
]


def _rng_array(shape: tuple[int, ...], seed: int) -> jax.Array:
    # Random values avoid ties in router logits, so the top-k order is unambiguous and the
    # independent argsort reference must match the traced indices exactly.
    return jnp.asarray(jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=jnp.float32))


def _linear(weights: Array, biases: Array | None, output_dims: tuple[int, ...], *, is_sharded: bool = True) -> Linear:
    return Linear(
        config=LinearConfig(),
        sharding_config=make_test_sharding_config(),
        weights=FullPrecisionSpec().compress(
            weights,
            sharding_config=make_test_sharding_config(),
            is_sharded=is_sharded,
        ),
        biases=biases,
        output_dims=output_dims,
    )


def _moe(*, num_shared_experts: int = 0, with_gate: bool = False) -> MixtureOfExperts:
    expert_config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    config = MixtureOfExpertsConfig(
        expert_config=expert_config,
        router_config=LinearConfig(),
        routing_function=SoftmaxRouting(),
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_active_routed_experts=NUM_ACTIVE,
        router_has_biases=True,
        num_shared_experts=num_shared_experts,
        expert_hidden_dim=HIDDEN_DIM,
        gate_config=LinearConfig() if with_gate else None,
    )
    mixture_size = NUM_ROUTED_EXPERTS + num_shared_experts
    return MixtureOfExperts(
        config=config,
        sharding_config=make_test_sharding_config(),
        router=_linear(
            _rng_array((NUM_ROUTED_EXPERTS, MODEL_DIM), seed=0),
            _rng_array((NUM_ROUTED_EXPERTS,), seed=1),
            (NUM_ROUTED_EXPERTS,),
        ),
        experts=DenseMLP(
            config=expert_config,
            sharding_config=make_test_sharding_config(),
            up_projection=_linear(
                _rng_array((mixture_size, 2 * HIDDEN_DIM, MODEL_DIM), seed=2),
                None,
                (HIDDEN_DIM, HIDDEN_DIM),
            ),
            down_projection=_linear(
                _rng_array((mixture_size, MODEL_DIM, HIDDEN_DIM), seed=3),
                None,
                (MODEL_DIM,),
            ),
        ),
        gate=_linear(_rng_array((1, MODEL_DIM), seed=4), None, (1,), is_sharded=False) if with_gate else None,
    )


def _reference_router_logits(module: MixtureOfExperts, inputs: Array) -> np.ndarray:
    # Independent reference: float64 numpy matmul over decompressed weights; does not call any
    # module forward path.
    weights = np.asarray(jax.device_get(module.router.weights.decompress()), dtype=np.float64)
    biases = np.asarray(jax.device_get(module.router.biases), dtype=np.float64)
    x = np.asarray(jax.device_get(inputs), dtype=np.float64)
    return x @ weights.T + biases


def _reference_topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    # Descending sort by value; ties are absent by construction (random continuous logits).
    return np.argsort(-logits, axis=-1)[..., :k]


def _trace_inputs(batch: int, tokens: int) -> Array:
    return jax.device_put(
        _rng_array((batch, tokens, MODEL_DIM), seed=10),
        make_sharding((LogicalAxis.BATCH, None, None)),
    )


@pytest.mark.usefixtures("fake_mesh")
def test_routing_trace_matches_independent_router_reference() -> None:
    # Expectation derived from the router equation logits = W x + b (float64 numpy reference).
    module = _moe()
    inputs = _trace_inputs(batch=2, tokens=5)

    trace = module.routing_trace(inputs, keychain=Keychain.init(7, sharding_config=make_test_sharding_config()))

    assert trace is not None
    reference_logits = _reference_router_logits(module, inputs)
    # fp32 forward vs float64 reference: mantissa of fp32 ~1.2e-7, dims are tiny, so 1e-5 is lax.
    np.testing.assert_allclose(np.asarray(jax.device_get(trace.router_logits)), reference_logits, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(trace.active_expert_indices)),
        _reference_topk_indices(reference_logits, NUM_ACTIVE),
    )
    assert trace.shared_expert_gate is None


@pytest.mark.parametrize("mode", MOE_MODES)
@pytest.mark.usefixtures("fake_mesh")
def test_routing_trace_indices_cover_experts_used_by_dispatch(mode: ForwardPassMode) -> None:
    # Metamorphic check against the dispatch itself: zeroing the traced experts' outputs is
    # impossible from outside, but the module output must equal the weighted sum over exactly the
    # traced experts (softmax over their logits) -- computed here with an independent float64
    # expert reference. Catches a trace that reports different experts than the dispatch uses.
    module = _moe()
    inputs = _trace_inputs(batch=2, tokens=4 if mode == ForwardPassMode.MULTI_TOKEN else 1)
    keychain = Keychain.init(7, sharding_config=make_test_sharding_config())

    outputs = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=mode, moe_chunk_size_ratio=0.5),
        keychain=keychain,
    )
    trace = module.routing_trace(inputs, keychain=keychain)
    assert trace is not None

    up = np.asarray(jax.device_get(module.experts.up_projection.weights.decompress()), dtype=np.float64)
    down = np.asarray(jax.device_get(module.experts.down_projection.weights.decompress()), dtype=np.float64)
    x = np.asarray(jax.device_get(inputs), dtype=np.float64)
    logits = np.asarray(jax.device_get(trace.router_logits), dtype=np.float64)
    indices = np.asarray(jax.device_get(trace.active_expert_indices))

    batch, tokens, _ = x.shape
    reference = np.zeros_like(x)
    for b in range(batch):
        for t in range(tokens):
            active = indices[b, t]
            active_logits = logits[b, t, active]
            weights = np.exp(active_logits - active_logits.max())
            weights /= weights.sum()
            for weight, expert in zip(weights, active, strict=True):
                # Identity activation, no biases: expert(x) = down @ (gate_half * up_half),
                # where up projection stacks [gate; up] halves.
                projected = up[expert] @ x[b, t]
                gate_half, up_half = projected[:HIDDEN_DIM], projected[HIDDEN_DIM:]
                reference[b, t] += weight * (down[expert] @ (gate_half * up_half))

    np.testing.assert_allclose(np.asarray(jax.device_get(outputs)), reference, atol=1e-4)


@pytest.mark.usefixtures("fake_mesh")
def test_routing_trace_shared_gate_matches_sigmoid_reference() -> None:
    # Expectation from the shared-expert gate equation: sigmoid(W_gate x) (float64 reference).
    # 2 shared experts keep mixture_size divisible by the 2-way mixture sharding axis.
    module = _moe(num_shared_experts=2, with_gate=True)
    inputs = _trace_inputs(batch=2, tokens=3)

    trace = module.routing_trace(inputs, keychain=Keychain.init(7, sharding_config=make_test_sharding_config()))

    assert trace is not None
    assert trace.shared_expert_gate is not None
    assert module.gate is not None
    gate_weights = np.asarray(jax.device_get(module.gate.weights.decompress()), dtype=np.float64)
    x = np.asarray(jax.device_get(inputs), dtype=np.float64)
    reference = 1.0 / (1.0 + np.exp(-(x @ gate_weights.T)))
    np.testing.assert_allclose(
        np.asarray(jax.device_get(trace.shared_expert_gate)),
        reference,
        atol=1e-5,
    )


@pytest.mark.usefixtures("fake_mesh")
def test_dense_mlp_routing_trace_is_none() -> None:
    # Contract: routing traces exist only for mixtures; dense MLPs return None.
    config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    module = DenseMLP(
        config=config,
        sharding_config=make_test_sharding_config(),
        up_projection=_linear(_rng_array((2 * HIDDEN_DIM, MODEL_DIM), seed=5), None, (HIDDEN_DIM, HIDDEN_DIM)),
        down_projection=_linear(_rng_array((MODEL_DIM, HIDDEN_DIM), seed=6), None, (MODEL_DIM,)),
    )
    inputs = _trace_inputs(batch=2, tokens=2)
    trace = module.routing_trace(inputs, keychain=Keychain.init(7, sharding_config=make_test_sharding_config()))
    assert trace is None


@pytest.mark.parametrize("mode", MOE_MODES)
@pytest.mark.usefixtures("fake_mesh")
def test_moe_accepts_per_sequence_keychain(mode: ForwardPassMode) -> None:
    # Regression: the ContinuousBatchScheduler hands the MLP a keychain with one key PER SEQUENCE
    # (vmapped_keys shape (batch,)); the MoE prefill path used to assume a scalar keychain and
    # crashed on keychain broadcasts (router flatten, chunk scan, shared-expert fan-out). The
    # deterministic inference forward consumes no randomness, so outputs must match the
    # scalar-keychain outputs exactly.
    module = _moe(num_shared_experts=2, with_gate=True)
    tokens = 4 if mode == ForwardPassMode.MULTI_TOKEN else 1
    inputs = _trace_inputs(batch=2, tokens=tokens)
    config = MLPForwardPassConfig(mode=mode, moe_chunk_size_ratio=0.5)

    scalar_out = module(
        inputs,
        forward_pass_config=config,
        keychain=Keychain.init(7, sharding_config=make_test_sharding_config()),
    )
    batched_out = module(
        inputs,
        forward_pass_config=config,
        keychain=Keychain.init(7, shape=(2,), sharding_config=make_test_sharding_config()),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(scalar_out)),
        np.asarray(jax.device_get(batched_out)),
    )
