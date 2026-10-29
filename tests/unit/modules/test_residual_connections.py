from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
import torch
import torch.nn.functional as torch_functional
from jaxtyping import Array

from lalamo.initializer import EmptyInitializer, RandomInitializer
from lalamo.module import Keychain
from lalamo.modules.flow import Context, FlowForwardPassConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.residual_connections import (
    AttentionResidual,
    AttentionResidualConfig,
    HyperConnection,
    HyperConnectionConfig,
    HyperConnectionDynamicsConfig,
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    Merge,
)
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

BATCH, TOKENS, STREAMS, CHANNELS = 2, 3, 4, 8
LAYER_NORM = NormalizationConfig(
    epsilon=1e-5,
    scale_offset=None,
    upcast_mode=UpcastMode.FULL_LAYER,
    subtract_mean=True,
    has_biases=True,
)
RMS_NORM = NormalizationConfig(
    epsilon=1e-6,
    scale_offset=None,
    upcast_mode=UpcastMode.FULL_LAYER,
    subtract_mean=False,
)
STATIC_HC_CONFIG = HyperConnectionConfig(STREAMS, read_stream=1, dynamics=None)
DYNAMIC_HC_CONFIG = HyperConnectionConfig(STREAMS, read_stream=1, dynamics=HyperConnectionDynamicsConfig(LAYER_NORM))
MHC_CONFIG = ManifoldHyperConnectionConfig(streams=STREAMS)
ATTENTION_RESIDUAL_CONFIG = AttentionResidualConfig(RMS_NORM)
HC_CONFIGS = [pytest.param(STATIC_HC_CONFIG, id="static-hc"), pytest.param(DYNAMIC_HC_CONFIG, id="dynamic-hc")]

type RoutedModule = HyperConnection | ManifoldHyperConnection
# A Lalamo module whose array leaves were replaced by float64 torch tensors, which the type system cannot express.
type TorchModule = Any


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


FLOAT32_TOLERANCE = Tolerance(atol=2e-5, rtol=2e-4)
BFLOAT16_TOLERANCE = Tolerance(atol=6e-2, rtol=4e-2)


def _initializer(*, seed: int = 0) -> RandomInitializer:
    return RandomInitializer(jnp.float32, make_test_sharding_config(), key=jax.random.key(seed))


def _trained[ModuleT: eqx.Module](module: ModuleT, *, seed: int) -> ModuleT:
    generator = np.random.default_rng(seed)
    return jtu.tree_map(
        lambda array: jnp.asarray(np.asarray(array) + generator.normal(0, 0.3, array.shape), array.dtype),
        module,
    )


def _random(shape: tuple[int, ...], *, seed: int, dtype: jnp.dtype = jnp.float32) -> Array:
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), dtype)


def _to_torch(array: Array) -> torch.Tensor:
    return torch.tensor(np.asarray(array.astype(jnp.float32), dtype=np.float64), requires_grad=True)


def _routed(module: RoutedModule, streams: Array) -> Array:
    routing = module.route(streams)
    return Merge().forward(
        routing,
        jnp.sin(routing.inputs),
        ctx=Context(
            None,
            jnp.zeros(streams.shape[:2], dtype=jnp.int32),
            None,
            None,
            FlowForwardPassConfig(),
            module.sharding_config,
        ),
        keychain=Keychain.init(0, sharding_config=module.sharding_config),
    )


def _assert_close(result: Array, expected: torch.Tensor, tolerance: Tolerance) -> None:
    np.testing.assert_allclose(
        np.asarray(result.astype(jnp.float32)),
        expected.detach().numpy(),
        atol=tolerance.atol,
        rtol=tolerance.rtol,
    )


def _assert_matches_torch[ModuleT: eqx.Module](
    call: Callable[..., Array],
    reference: Callable[..., torch.Tensor],
    module: ModuleT,
    *inputs: Array,
) -> None:
    torch_module = jtu.tree_map(_to_torch, module)
    torch_inputs = tuple(_to_torch(array) for array in inputs)
    expected = reference(torch_module, *torch_inputs)
    result = jax.jit(call)(module, *inputs)
    assert result.dtype == inputs[0].dtype
    if result.dtype == jnp.bfloat16:
        _assert_close(result, expected, BFLOAT16_TOLERANCE)
        return
    _assert_close(result, expected, FLOAT32_TOLERANCE)

    cotangent = _random(result.shape, seed=99)
    (expected * _to_torch(cotangent).detach()).sum().backward()
    gradients = jax.jit(
        jax.grad(lambda *args: jnp.sum(call(*args) * cotangent), argnums=tuple(range(len(inputs) + 1)))
    )(module, *inputs)
    expected_gradients = jtu.tree_leaves((torch_module, torch_inputs))
    actual_gradients = jtu.tree_leaves(gradients)
    assert len(actual_gradients) == len(expected_gradients)
    for actual, expected_tensor in zip(actual_gradients, expected_gradients, strict=True):
        assert expected_tensor.grad is not None
        _assert_close(actual, expected_tensor.grad, FLOAT32_TOLERANCE)


def _torch_hyper_connection(module: TorchModule, streams: torch.Tensor) -> torch.Tensor:
    hidden = streams.unflatten(-1, (STREAMS, CHANNELS))
    alpha, beta = module.static_alpha, module.static_beta
    if module.dynamics is not None:
        norm = module.dynamics.norm
        normalized = torch_functional.layer_norm(
            hidden, (CHANNELS,), norm.scales, norm.biases, eps=norm.config.epsilon
        )
        alpha = alpha + module.dynamics.alpha_scale * torch.tanh(normalized @ module.dynamics.alpha_projection.T)
        beta = beta + module.dynamics.beta_scale * torch.tanh(normalized @ module.dynamics.beta_projection)
    mixed = alpha.transpose(-1, -2) @ hidden
    output = mixed[..., 1:, :] + beta.unsqueeze(-1) * torch.sin(mixed[..., 0, :]).unsqueeze(-2)
    return output.flatten(-2)


def _torch_manifold_hyper_connection(module: TorchModule, streams: torch.Tensor) -> torch.Tensor:
    config = module.config
    hidden = streams.unflatten(-1, (STREAMS, CHANNELS))
    normalized = streams * torch.rsqrt(streams.square().mean(-1, keepdim=True) + config.norm_epsilon)
    pre = torch.sigmoid(normalized @ module.pre_projection.T * module.pre_scale + module.pre_bias) + config.epsilon
    post = 2 * torch.sigmoid(normalized @ module.post_projection.T * module.post_scale + module.post_bias)
    residual_logits = (
        torch.einsum("...w,stw->...st", normalized, module.residual_projection) * module.residual_scale
        + module.residual_bias
    )
    mixing = torch.softmax(residual_logits, dim=-1) + config.epsilon
    mixing = mixing / (mixing.sum(dim=-2, keepdim=True) + config.epsilon)
    for _ in range(config.sinkhorn_iterations - 1):
        mixing = mixing / (mixing.sum(dim=-1, keepdim=True) + config.epsilon)
        mixing = mixing / (mixing.sum(dim=-2, keepdim=True) + config.epsilon)
    collapsed = (pre.unsqueeze(-1) * hidden).sum(dim=-2)
    output = mixing.transpose(-1, -2) @ hidden + post.unsqueeze(-1) * torch.sin(collapsed).unsqueeze(-2)
    return output.flatten(-2)


def _torch_attention_residual(
    module: TorchModule,
    current: torch.Tensor,
    *previous: torch.Tensor,
) -> torch.Tensor:
    values = torch.stack((*previous, current))
    keys = (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + module.key_norm.config.epsilon)
        * module.key_norm.scales
    )
    logits = keys @ module.query
    if module.recency_bias is not None:
        logits = torch.cat((logits[:-1], logits[-1:] + module.recency_bias))
    weights = torch.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * values).sum(dim=0)


def _attention_residual(module: AttentionResidual, current: Array, *previous: Array) -> Array:
    return module(current, *previous)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("config", HC_CONFIGS)
def test_hyper_connection_matches_paper_equations(config: HyperConnectionConfig, dtype: jnp.dtype) -> None:
    module = _trained(config.init(_initializer(), CHANNELS), seed=1).astype(dtype)
    streams = _random((BATCH, TOKENS, STREAMS * CHANNELS), seed=2, dtype=dtype)
    _assert_matches_torch(_routed, _torch_hyper_connection, module, streams)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("sinkhorn_iterations", [1, 20])
def test_manifold_hyper_connection_matches_reference(dtype: jnp.dtype, sinkhorn_iterations: int) -> None:
    config = replace(MHC_CONFIG, sinkhorn_iterations=sinkhorn_iterations)
    module = _trained(config.init(_initializer(), CHANNELS), seed=3).astype(dtype)
    streams = _random((BATCH, TOKENS, STREAMS * CHANNELS), seed=4, dtype=dtype)
    _assert_matches_torch(_routed, _torch_manifold_hyper_connection, module, streams)


@pytest.mark.parametrize("num_previous", [0, 1, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("recency_bias_init", [None, 0.2])
def test_attention_residual_matches_reference(
    num_previous: int, dtype: jnp.dtype, recency_bias_init: float | None
) -> None:
    config = replace(ATTENTION_RESIDUAL_CONFIG, recency_bias_init=recency_bias_init)
    module = _trained(config.init(_initializer(), CHANNELS), seed=5).astype(dtype)
    current, *previous = (
        _random((BATCH, TOKENS, CHANNELS), seed=6 + index, dtype=dtype) for index in range(num_previous + 1)
    )
    _assert_matches_torch(_attention_residual, _torch_attention_residual, module, current, *previous)


def test_manifold_residual_mixing_is_doubly_stochastic() -> None:
    module = _trained(MHC_CONFIG.init(_initializer(), CHANNELS), seed=7)
    shared_stream = _random((BATCH, TOKENS, 1, CHANNELS), seed=8)
    identical_streams = jnp.broadcast_to(shared_stream, (BATCH, TOKENS, STREAMS, CHANNELS))
    column_routing = module.route(identical_streams.reshape(BATCH, TOKENS, -1))
    np.testing.assert_allclose(
        column_routing.residual.reshape(identical_streams.shape),
        identical_streams,
        atol=1e-5,
        rtol=1e-5,
    )

    streams = _random((BATCH, TOKENS, STREAMS * CHANNELS), seed=9)
    routing = module.route(streams)
    np.testing.assert_allclose(
        routing.residual.reshape(BATCH, TOKENS, STREAMS, CHANNELS).sum(axis=-2),
        streams.reshape(BATCH, TOKENS, STREAMS, CHANNELS).sum(axis=-2),
        atol=1e-3,
        rtol=1e-3,
    )
    assert jnp.all((routing.output_weights > 0) & (routing.output_weights < 2))
    assert {routing.inputs.dtype, routing.residual.dtype, routing.output_weights.dtype} == {jnp.dtype(jnp.float32)}


def test_mhc_rounds_only_the_final_stream_update() -> None:
    module = _trained(MHC_CONFIG.init(_initializer(), CHANNELS), seed=17)
    streams = _random((BATCH, TOKENS, STREAMS * CHANNELS), seed=18, dtype=jnp.bfloat16)
    routing = module.route(streams)
    assert routing.residual.dtype == routing.output_weights.dtype == jnp.float32
    contribution = np.asarray(jnp.sin(routing.inputs).astype(jnp.float32))
    residual = np.asarray(routing.residual).reshape(BATCH, TOKENS, STREAMS, CHANNELS)
    expected = residual + np.asarray(routing.output_weights)[..., None] * contribution[..., None, :]
    expected = jnp.asarray(expected.reshape(streams.shape), dtype=streams.dtype)
    actual = jax.jit(_routed)(module, streams)
    np.testing.assert_array_equal(np.asarray(actual.astype(jnp.float32)), np.asarray(expected.astype(jnp.float32)))


@pytest.mark.parametrize("config", HC_CONFIGS)
def test_hyper_connection_initializes_as_canonical_residual(config: HyperConnectionConfig) -> None:
    module = config.init(_initializer(), CHANNELS)
    streams = _random((BATCH, TOKENS, STREAMS * CHANNELS), seed=10)
    routing = module.route(streams)
    np.testing.assert_allclose(routing.inputs, streams.reshape(BATCH, TOKENS, STREAMS, CHANNELS)[..., 1, :])
    np.testing.assert_allclose(routing.residual, streams)
    np.testing.assert_allclose(routing.output_weights, jnp.ones((BATCH, TOKENS, STREAMS)))


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(STATIC_HC_CONFIG, id="static-hc"),
        pytest.param(DYNAMIC_HC_CONFIG, id="dynamic-hc"),
        pytest.param(MHC_CONFIG, id="mhc"),
        pytest.param(ATTENTION_RESIDUAL_CONFIG, id="attention-residual"),
    ],
)
def test_export_round_trips_into_empty_skeleton(
    config: HyperConnectionConfig | ManifoldHyperConnectionConfig | AttentionResidualConfig,
) -> None:
    trained = _trained(config.init(_initializer(), CHANNELS), seed=11)
    skeleton = config.init(EmptyInitializer(jnp.float32, make_test_sharding_config()), CHANNELS)
    restored = skeleton.load_exported(trained.export())
    for restored_array, trained_array in zip(jtu.tree_leaves(restored), jtu.tree_leaves(trained), strict=True):
        np.testing.assert_array_equal(restored_array, trained_array)
