from math import prod

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.activations import Identity, SiLU
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
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config

MODEL_DIM = 4
HIDDEN_DIM = 4
NUM_ROUTED_EXPERTS = 2

MOE_MODES = [
    pytest.param(ForwardPassMode.MULTI_TOKEN, id="multi-token"),
    pytest.param(ForwardPassMode.SINGLE_TOKEN, id="single-token"),
]


def _array(shape: tuple[int, ...], *, offset: int = 0) -> jax.Array:
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 10) - 1


def _moe_array(shape: tuple[int, ...], *, offset: int = 0) -> jax.Array:
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 50) - 0.5


def _linear(weights: Array, biases: Array | None, output_dims: tuple[int, ...]) -> Linear:
    return Linear(
        config=LinearConfig(),
        sharding_config=make_test_sharding_config(),
        weights=FullPrecisionSpec().compress(weights, sharding_config=make_test_sharding_config()),
        biases=biases,
        output_dims=output_dims,
    )


def _dense_mlp(config: DenseMLPConfig | None = None) -> DenseMLP:
    if config is None:
        config = DenseMLPConfig(
            linear_config=LinearConfig(),
            activation=SiLU(),
            has_up_biases=True,
            has_down_biases=True,
            gate_clipping=(-0.5, 0.75),
            up_clipping=(-0.25, 0.5),
        )
    return DenseMLP(
        config=config,
        sharding_config=make_test_sharding_config(),
        up_projection=_linear(
            _array((2 * HIDDEN_DIM, MODEL_DIM)),
            _array((2 * HIDDEN_DIM,), offset=100) if config.has_up_biases else None,
            (HIDDEN_DIM, HIDDEN_DIM),
        ),
        down_projection=_linear(
            _array((MODEL_DIM, HIDDEN_DIM), offset=200),
            _array((MODEL_DIM,), offset=300) if config.has_down_biases else None,
            (MODEL_DIM,),
        ),
    )


def _moe(
    *,
    num_active_routed_experts: int = 1,
    num_shared_experts: int = 0,
) -> MixtureOfExperts:
    expert_config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=Identity(),
        has_up_biases=True,
        has_down_biases=True,
        gate_clipping=None,
        up_clipping=None,
    )
    config = MixtureOfExpertsConfig(
        expert_config=expert_config,
        router_config=LinearConfig(),
        routing_function=SoftmaxRouting(),
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_active_routed_experts=num_active_routed_experts,
        router_has_biases=True,
        num_shared_experts=num_shared_experts,
        expert_hidden_dim=HIDDEN_DIM,
        gate_config=None,
    )
    mixture_size = NUM_ROUTED_EXPERTS + num_shared_experts
    expert_bias_sharding = make_sharding((None, None))
    return MixtureOfExperts(
        config=config,
        sharding_config=make_test_sharding_config(),
        router=_linear(
            _moe_array((NUM_ROUTED_EXPERTS, MODEL_DIM)),
            _moe_array((NUM_ROUTED_EXPERTS,), offset=100),
            (NUM_ROUTED_EXPERTS,),
        ),
        experts=DenseMLP(
            config=expert_config,
            sharding_config=make_test_sharding_config(),
            up_projection=_linear(
                _moe_array((mixture_size, 2 * HIDDEN_DIM, MODEL_DIM), offset=200),
                jax.device_put(_moe_array((mixture_size, 2 * HIDDEN_DIM), offset=300), expert_bias_sharding),
                (HIDDEN_DIM, HIDDEN_DIM),
            ),
            down_projection=_linear(
                _moe_array((mixture_size, MODEL_DIM, HIDDEN_DIM), offset=400),
                jax.device_put(_moe_array((mixture_size, MODEL_DIM), offset=500), expert_bias_sharding),
                (MODEL_DIM,),
            ),
        ),
        gate=None,
    )


def _reference(module: DenseMLP, inputs: Array) -> Array:
    inputs = jnp.asarray(jax.device_get(inputs))
    up_weights = jnp.asarray(jax.device_get(module.up_projection.weights.decompress()))
    down_weights = jnp.asarray(jax.device_get(module.down_projection.weights.decompress()))
    up_result = jnp.einsum("...i,oi->...o", inputs, up_weights)
    if module.up_projection.biases is not None:
        up_result = up_result + jnp.asarray(jax.device_get(module.up_projection.biases))
    up_projection, gate = jnp.split(up_result, (HIDDEN_DIM,), axis=-1)
    if module.config.gate_clipping is not None:
        gate = jnp.clip(gate, *module.config.gate_clipping)
    if module.config.up_clipping is not None:
        up_projection = jnp.clip(up_projection, *module.config.up_clipping)
    hidden = up_projection * module.config.activation(gate)
    result = jnp.einsum("...i,oi->...o", hidden, down_weights)
    if module.down_projection.biases is not None:
        result = result + jnp.asarray(jax.device_get(module.down_projection.biases))
    return result


def _router_logits_reference(module: MixtureOfExperts, inputs: Array) -> Array:
    router_weights = jnp.asarray(jax.device_get(module.router.weights.decompress()))
    logits = router_weights @ inputs
    if module.router.biases is not None:
        logits = logits + jnp.asarray(jax.device_get(module.router.biases))
    return logits


def _expert_reference(module: MixtureOfExperts, expert_idx: int, inputs: Array) -> Array:
    up_weights = jnp.asarray(jax.device_get(module.experts.up_projection.weights.decompress()))[expert_idx]
    down_weights = jnp.asarray(jax.device_get(module.experts.down_projection.weights.decompress()))[expert_idx]
    up_result = up_weights @ inputs
    if module.experts.up_projection.biases is not None:
        up_biases = jnp.asarray(jax.device_get(module.experts.up_projection.biases))
        up_result = up_result + up_biases[expert_idx]
    up_projection, gate = jnp.split(up_result, (module.experts.hidden_dim,))
    hidden = up_projection * module.experts.config.activation(gate)
    result = down_weights @ hidden
    if module.experts.down_projection.biases is not None:
        down_biases = jnp.asarray(jax.device_get(module.experts.down_projection.biases))
        result = result + down_biases[expert_idx]
    return result


def _routed_moe_token_reference(module: MixtureOfExperts, inputs: Array) -> Array:
    logits = _router_logits_reference(module, inputs)
    active_logits, active_indices = jax.lax.top_k(logits, module.config.num_active_routed_experts)
    active_weights = jax.nn.softmax(active_logits)
    result = jnp.zeros((module.model_dim,), dtype=inputs.dtype)
    for active_idx, active_weight in zip(jax.device_get(active_indices), active_weights, strict=True):
        result = result + _expert_reference(module, int(active_idx), inputs) * active_weight
    for expert_idx in range(module.config.num_routed_experts, module.config.mixture_size):
        result = result + _expert_reference(module, expert_idx, inputs)
    return result


def _routed_moe_reference(
    module: MixtureOfExperts,
    inputs: Array,
    lengths_without_padding: Array | None = None,
) -> Array:
    host_inputs = jnp.asarray(jax.device_get(inputs))
    host_lengths = jax.device_get(lengths_without_padding) if lengths_without_padding is not None else None
    return jnp.stack(
        [
            jnp.stack(
                [
                    _routed_moe_token_reference(module, token_inputs)
                    if host_lengths is None or token_idx < host_lengths[batch_idx]
                    else jnp.zeros((module.model_dim,), dtype=token_inputs.dtype)
                    for token_idx, token_inputs in enumerate(batch_inputs)
                ]
            )
            for batch_idx, batch_inputs in enumerate(host_inputs)
        ],
    )


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _sharded_vector(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None,)))


def _sharded_tokens(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.BATCH, None, None)))


def _sharded_lengths(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.BATCH,)))


def _moe_sequence_length(mode: ForwardPassMode) -> int:
    if mode == ForwardPassMode.SINGLE_TOKEN:
        return 1
    return 3


def test_dense_mlp_call_unbatched_matches_reference_and_keeps_unsharded_features(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_vector(jnp.arange(MODEL_DIM, dtype=jnp.float32))

    result = module.call_unbatched(inputs, keychain=Keychain.init(0, sharding_config=make_test_sharding_config()))

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_softmax_routing_call_unbatched_selects_top_k_and_normalizes_weights() -> None:
    routing = SoftmaxRouting()
    logits = jnp.array([1.0, 3.0, 2.0], dtype=jnp.float32)

    result = routing.call_unbatched(logits, num_active=2)

    expected_weights = jnp.zeros_like(logits)
    expected_weights = expected_weights.at[jnp.array([1, 2])].set(jax.nn.softmax(jnp.array([3.0, 2.0])))
    assert jnp.array_equal(result.expert_mask, jnp.array([False, True, True]))
    assert_close(result=result.expert_weights, reference=expected_weights)
    assert_close(result=jnp.sum(result.expert_weights), reference=jnp.array(1.0, dtype=jnp.float32))


@pytest.mark.parametrize("mode", MOE_MODES)
@pytest.mark.usefixtures("fake_mesh")
def test_routed_moe_matches_direct_reference(mode: ForwardPassMode) -> None:
    module = _moe(num_active_routed_experts=2)
    sequence_length = _moe_sequence_length(mode)
    inputs = _sharded_tokens(
        jnp.arange(2 * sequence_length * MODEL_DIM, dtype=jnp.float32).reshape(
            2,
            sequence_length,
            MODEL_DIM,
        )
        / 10
    )

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=mode, moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(7, sharding_config=make_test_sharding_config()),
    )

    _assert_close(result=result, reference=_routed_moe_reference(module, inputs))


@pytest.mark.usefixtures("fake_mesh")
def test_moe_prefill_with_shared_experts_and_padding_matches_direct_reference() -> None:
    module = _moe(num_active_routed_experts=2, num_shared_experts=2)
    inputs = _sharded_tokens(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)
    lengths_without_padding = _sharded_lengths(jnp.array([3, 1], dtype=jnp.int32))

    result = module(
        inputs,
        lengths_without_padding=lengths_without_padding,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(9, sharding_config=make_test_sharding_config()),
    )

    _assert_close(
        result=result,
        reference=_routed_moe_reference(module, inputs, lengths_without_padding=lengths_without_padding),
    )
