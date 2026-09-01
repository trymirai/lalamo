from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.compressed import IntSpec
from lalamo.initializer import RandomInitializer
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
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, WeightMatrixSpec
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config

MODEL_DIM = 4
HIDDEN_DIM = 4
NUM_ROUTED_EXPERTS = 4

MOE_MODES = [
    pytest.param(ForwardPassMode.MULTI_TOKEN, id="multi-token"),
    pytest.param(ForwardPassMode.SINGLE_TOKEN, id="single-token"),
]


def _array(shape: tuple[int, ...], *, offset: int = 0) -> jax.Array:
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 10) - 1


def _moe_array(shape: tuple[int, ...], *, offset: int = 0) -> jax.Array:
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 50) - 0.5


def _linear(
    weights: Array,
    biases: Array | None,
    output_dims: tuple[int, ...],
    weight_spec: WeightMatrixSpec = FullPrecisionSpec(),
    *,
    is_sharded: bool = True,
) -> Linear:
    return Linear(
        config=LinearConfig(),
        sharding_config=make_test_sharding_config(),
        weights=weight_spec.compress(
            weights,
            sharding_config=make_test_sharding_config(),
            is_sharded=is_sharded,
        ),
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


def _expert_mlp(
    config: DenseMLPConfig,
    mixture_size: int,
    weight_spec: WeightMatrixSpec,
    *,
    offset: int,
    is_sharded: bool,
) -> DenseMLP:
    expert_bias_sharding = make_sharding((None, None))
    return DenseMLP(
        config=config,
        sharding_config=make_test_sharding_config(),
        up_projection=_linear(
            _moe_array((mixture_size, 2 * HIDDEN_DIM, MODEL_DIM), offset=offset),
            jax.device_put(_moe_array((mixture_size, 2 * HIDDEN_DIM), offset=offset + 100), expert_bias_sharding),
            (HIDDEN_DIM, HIDDEN_DIM),
            weight_spec,
            is_sharded=is_sharded,
        ),
        down_projection=_linear(
            _moe_array((mixture_size, MODEL_DIM, HIDDEN_DIM), offset=offset + 200),
            jax.device_put(_moe_array((mixture_size, MODEL_DIM), offset=offset + 300), expert_bias_sharding),
            (MODEL_DIM,),
            weight_spec,
            is_sharded=is_sharded,
        ),
    )


def _moe(
    *,
    num_active_routed_experts: int = 1,
    num_shared_experts: int = 0,
    use_shared_gate: bool = False,
    expert_weight_spec: WeightMatrixSpec = FullPrecisionSpec(),
    routed_experts_sharded: bool = False,
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
        gate_config=LinearConfig() if use_shared_gate else None,
    )
    return MixtureOfExperts(
        config=config,
        sharding_config=make_test_sharding_config(),
        router=_linear(
            _moe_array((NUM_ROUTED_EXPERTS, MODEL_DIM)),
            _moe_array((NUM_ROUTED_EXPERTS,), offset=100),
            (NUM_ROUTED_EXPERTS,),
            is_sharded=False,
        ),
        routed_experts=_expert_mlp(
            expert_config,
            NUM_ROUTED_EXPERTS,
            expert_weight_spec,
            offset=200,
            is_sharded=routed_experts_sharded,
        ),
        shared_experts=(
            _expert_mlp(
                expert_config,
                num_shared_experts,
                expert_weight_spec,
                offset=700,
                is_sharded=False,
            )
            if num_shared_experts > 0
            else None
        ),
        gate=(
            _linear(
                _moe_array((1, MODEL_DIM), offset=600),
                None,
                (1,),
                is_sharded=False,
            )
            if use_shared_gate
            else None
        ),
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


def _expert_reference(module: DenseMLP, expert_idx: int, inputs: Array) -> Array:
    up_weights = jnp.asarray(jax.device_get(module.up_projection.weights.decompress()))[expert_idx]
    down_weights = jnp.asarray(jax.device_get(module.down_projection.weights.decompress()))[expert_idx]
    up_result = up_weights @ inputs
    if module.up_projection.biases is not None:
        up_biases = jnp.asarray(jax.device_get(module.up_projection.biases))
        up_result = up_result + up_biases[expert_idx]
    up_projection, gate = jnp.split(up_result, (module.hidden_dim,))
    hidden = up_projection * module.config.activation(gate)
    result = down_weights @ hidden
    if module.down_projection.biases is not None:
        down_biases = jnp.asarray(jax.device_get(module.down_projection.biases))
        result = result + down_biases[expert_idx]
    return result


def _routed_moe_token_reference(module: MixtureOfExperts, inputs: Array) -> Array:
    logits = _router_logits_reference(module, inputs)
    active_logits, active_indices = jax.lax.top_k(logits, module.config.num_active_routed_experts)
    active_weights = jax.nn.softmax(active_logits)
    result = jnp.zeros((module.model_dim,), dtype=inputs.dtype)
    for active_idx, active_weight in zip(jax.device_get(active_indices), active_weights, strict=True):
        result = result + _expert_reference(module.routed_experts, int(active_idx), inputs) * active_weight
    shared_weight = jnp.array(1.0, dtype=inputs.dtype)
    if module.gate is not None:
        gate_weights = jnp.asarray(jax.device_get(module.gate.weights.decompress()))
        shared_weight = jax.nn.sigmoid((gate_weights @ inputs).squeeze())
    if module.shared_experts is not None:
        for expert_idx in range(module.config.num_shared_experts):
            result = result + _expert_reference(module.shared_experts, expert_idx, inputs) * shared_weight
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


def _differentiable_moe_reference(
    module: MixtureOfExperts,
    inputs: Array,
    lengths_without_padding: Array,
) -> Array:
    def call_experts(experts: DenseMLP) -> Array:
        up_weights = experts.up_projection.weights
        down_weights = experts.down_projection.weights
        assert isinstance(up_weights, FullPrecisionMatrix)
        assert isinstance(down_weights, FullPrecisionMatrix)
        up_result = jnp.einsum("bti,eoi->bteo", inputs, up_weights.decompress())
        if experts.up_projection.biases is not None:
            up_result = up_result + experts.up_projection.biases[None, None]
        up_projection, gate = jnp.split(up_result, (experts.hidden_dim,), axis=-1)
        if experts.config.gate_clipping is not None:
            gate = jnp.clip(gate, *experts.config.gate_clipping)
        if experts.config.up_clipping is not None:
            up_projection = jnp.clip(up_projection, *experts.config.up_clipping)
        hidden = up_projection * experts.config.activation(gate)
        result = jnp.einsum("bteh,eoh->bteo", hidden, down_weights.decompress())
        if experts.down_projection.biases is not None:
            result = result + experts.down_projection.biases[None, None]
        return result

    router_weights = module.router.weights
    assert isinstance(router_weights, FullPrecisionMatrix)
    router_logits = jnp.einsum("bti,ei->bte", inputs, router_weights.decompress())
    if module.router.biases is not None:
        router_logits = router_logits + module.router.biases
    active_logits, active_indices = jax.lax.top_k(router_logits, module.config.num_active_routed_experts)
    active_weights = jax.nn.softmax(active_logits)
    routed_outputs = call_experts(module.routed_experts)
    active_outputs = jnp.take_along_axis(routed_outputs, active_indices[..., None], axis=2)
    result = jnp.sum(active_outputs * active_weights[..., None], axis=2)

    if module.shared_experts is not None:
        shared_weight = jnp.ones((*inputs.shape[:2], 1), dtype=inputs.dtype)
        if module.gate is not None:
            gate_weights = module.gate.weights
            assert isinstance(gate_weights, FullPrecisionMatrix)
            shared_weight = jax.nn.sigmoid(jnp.einsum("bti,oi->bto", inputs, gate_weights.decompress()))
        result = result + shared_weight * call_experts(module.shared_experts).sum(axis=2)

    padding_mask = jnp.arange(inputs.shape[1])[None, :] < lengths_without_padding[:, None]
    return jnp.where(padding_mask[..., None], result, 0.0)


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

    assert jnp.array_equal(result.active_expert_indices, jnp.array([1, 2]))
    assert_close(result=result.active_expert_weights, reference=jax.nn.softmax(jnp.array([3.0, 2.0])))
    assert_close(result=jnp.sum(result.active_expert_weights), reference=jnp.array(1.0, dtype=jnp.float32))


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


def test_moe_config_splits_sharded_routed_and_replicated_shared_experts(fake_mesh: Mesh) -> None:
    template = _moe(num_shared_experts=1, use_shared_gate=True)
    module = template.config.init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
            key=jax.random.key(20),
        ),
        MODEL_DIM,
        HIDDEN_DIM,
    )

    assert module.shared_experts is not None
    routed_weights = module.routed_experts.up_projection.weights
    shared_weights = module.shared_experts.up_projection.weights
    assert isinstance(routed_weights, FullPrecisionMatrix)
    assert isinstance(shared_weights, FullPrecisionMatrix)
    assert routed_weights.weights.sharding == make_sharding(
        (LogicalAxis.MIXTURE, None, None),
    )
    assert shared_weights.weights.sharding == make_sharding((None, None, None))
    assert module.gate is not None
    gate_weights = module.gate.weights
    assert isinstance(gate_weights, FullPrecisionMatrix)
    assert gate_weights.weights.sharding == make_sharding((None, None))
    _assert_named_sharding(routed_weights.weights.sharding, fake_mesh)


@pytest.mark.parametrize("mode", MOE_MODES)
@pytest.mark.usefixtures("fake_mesh")
def test_full_precision_routed_moe_preserves_input_dtype(mode: ForwardPassMode) -> None:
    module = _moe(num_active_routed_experts=2)
    sequence_length = _moe_sequence_length(mode)
    inputs = _sharded_tokens(
        jnp.arange(2 * sequence_length * MODEL_DIM, dtype=jnp.bfloat16).reshape(
            2,
            sequence_length,
            MODEL_DIM,
        )
        / 10,
    )

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=mode),
        keychain=Keychain.init(8, sharding_config=make_test_sharding_config()),
    )

    assert result.dtype == inputs.dtype


@pytest.mark.parametrize("length_values", [(3, 1), (0, 0)], ids=["partial-padding", "all-padding"])
@pytest.mark.usefixtures("fake_mesh")
def test_full_precision_moe_prefill_with_gated_shared_experts_and_padding_matches_reference(
    length_values: tuple[int, int],
) -> None:
    module = _moe(num_active_routed_experts=1, num_shared_experts=1, use_shared_gate=True)
    inputs = _sharded_tokens(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)
    lengths_without_padding = _sharded_lengths(jnp.array(length_values, dtype=jnp.int32))

    result = module(
        inputs,
        lengths_without_padding=lengths_without_padding,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.0),
        keychain=Keychain.init(9, sharding_config=make_test_sharding_config()),
    )

    _assert_close(
        result=result,
        reference=_routed_moe_reference(module, inputs, lengths_without_padding=lengths_without_padding),
    )
    host_result = jax.device_get(result)
    padding_mask = jnp.arange(3)[None, :] >= jnp.array(length_values)[:, None]
    assert jnp.array_equal(host_result[padding_mask], jnp.zeros((padding_mask.sum(), MODEL_DIM), dtype=result.dtype))


@pytest.mark.usefixtures("fake_mesh")
def test_full_precision_moe_training_gradients_match_dense_reference() -> None:
    module = _moe(num_active_routed_experts=1, num_shared_experts=1, use_shared_gate=True)
    inputs = _sharded_tokens(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)
    lengths_without_padding = _sharded_lengths(jnp.array([3, 1], dtype=jnp.int32))

    def loss(module_and_inputs: tuple[MixtureOfExperts, Array]) -> Array:
        current_module, current_inputs = module_and_inputs
        outputs = current_module(
            current_inputs,
            lengths_without_padding=lengths_without_padding,
            forward_pass_config=MLPForwardPassConfig.for_training(),
            keychain=Keychain.init(10, sharding_config=make_test_sharding_config()),
        )
        return jnp.square(outputs).sum()

    def reference_loss(module_and_inputs: tuple[MixtureOfExperts, Array]) -> Array:
        current_module, current_inputs = module_and_inputs
        outputs = _differentiable_moe_reference(
            current_module,
            current_inputs,
            lengths_without_padding,
        )
        return jnp.square(outputs).sum()

    module_grad, input_grad = eqx.filter_grad(loss)((module, inputs))
    reference_module_grad, reference_input_grad = eqx.filter_grad(reference_loss)((module, inputs))

    array_grads = [leaf for leaf in jax.tree_util.tree_leaves(module_grad) if eqx.is_array(leaf)]
    reference_array_grads = [leaf for leaf in jax.tree_util.tree_leaves(reference_module_grad) if eqx.is_array(leaf)]
    assert array_grads
    assert len(array_grads) == len(reference_array_grads)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in array_grads)
    for gradient, reference_gradient in zip(array_grads, reference_array_grads, strict=True):
        assert_close(result=gradient, reference=reference_gradient)
    host_input_grad = jax.device_get(input_grad)
    assert jnp.all(jnp.isfinite(host_input_grad))
    assert_close(result=input_grad, reference=reference_input_grad)
    assert jnp.array_equal(host_input_grad[1, 1:], jnp.zeros((2, MODEL_DIM), dtype=host_input_grad.dtype))


@pytest.mark.usefixtures("fake_mesh")
def test_gated_shared_quantized_moe_decode_matches_reference() -> None:
    module = _moe(
        num_active_routed_experts=1,
        num_shared_experts=1,
        use_shared_gate=True,
        expert_weight_spec=IntSpec(bits=8, group_size=2),
    )
    inputs = _sharded_tokens(jnp.arange(2 * MODEL_DIM, dtype=jnp.float32).reshape(2, 1, MODEL_DIM) / 10)

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.SINGLE_TOKEN),
        keychain=Keychain.init(10, sharding_config=make_test_sharding_config()),
    )

    _assert_close(result=result, reference=_routed_moe_reference(module, inputs))


@pytest.mark.usefixtures("fake_mesh")
def test_quantized_moe_prefill_keeps_chunked_fallback() -> None:
    module = _moe(
        num_active_routed_experts=1,
        expert_weight_spec=IntSpec(bits=8, group_size=2),
    )
    inputs = _sharded_tokens(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)
    lengths_without_padding = _sharded_lengths(jnp.array([3, 1], dtype=jnp.int32))

    result = module(
        inputs,
        lengths_without_padding=lengths_without_padding,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(11, sharding_config=make_test_sharding_config()),
    )

    _assert_close(
        result=result,
        reference=_routed_moe_reference(module, inputs, lengths_without_padding=lengths_without_padding),
    )


@pytest.mark.usefixtures("fake_mesh")
def test_expert_sharded_full_precision_moe_prefill_keeps_chunked_fallback() -> None:
    module = _moe(
        num_active_routed_experts=1,
        routed_experts_sharded=True,
    )
    inputs = _sharded_tokens(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)
    lengths_without_padding = _sharded_lengths(jnp.array([3, 1], dtype=jnp.int32))

    result = module(
        inputs,
        lengths_without_padding=lengths_without_padding,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(12, sharding_config=make_test_sharding_config()),
    )

    _assert_close(
        result=result,
        reference=_routed_moe_reference(module, inputs, lengths_without_padding=lengths_without_padding),
    )
