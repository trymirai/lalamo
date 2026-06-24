from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.compressed.awq import AWQSpec
from lalamo.initializer import RandomInitializer
from lalamo.module import ForwardPassMode, Keychain, ShardingAxis
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
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, FullPrecisionMatrix, FullPrecisionSpec
from tests.common import assert_close

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


def _linear(
    weights: Array,
    biases: Array | None,
    output_dims: tuple[int, ...],
    *,
    is_sharded: bool = True,
) -> Linear:
    return Linear(
        config=LinearConfig(),
        weights=FullPrecisionSpec().compress(weights, is_sharded=is_sharded),
        biases=biases,
        output_dims=output_dims,
    )


def _linear_template(
    weights_shape: tuple[int, ...],
    biases_shape: tuple[int, ...] | None,
    output_dims: tuple[int, ...],
    bias_partition: tuple[ShardingAxis | None, ...] | None,
    *,
    is_sharded: bool = True,
) -> Linear:
    bias_sharding = make_sharding(bias_partition)
    return Linear(
        config=LinearConfig(),
        weights=FullPrecisionSpec().compress(dummy_array(weights_shape, jnp.float32), is_sharded=is_sharded),
        biases=dummy_array(biases_shape, jnp.float32, bias_sharding) if biases_shape is not None else None,
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
    expert_config: DenseMLPConfig,
    mixture_size: int,
    *,
    offset: int,
    is_sharded: bool,
) -> DenseMLP:
    expert_bias_sharding = make_sharding((None, None))
    return DenseMLP(
        config=expert_config,
        up_projection=_linear(
            _moe_array((mixture_size, 2 * HIDDEN_DIM, MODEL_DIM), offset=offset),
            jax.device_put(_moe_array((mixture_size, 2 * HIDDEN_DIM), offset=offset + 100), expert_bias_sharding),
            (HIDDEN_DIM, HIDDEN_DIM),
            is_sharded=is_sharded,
        ),
        down_projection=_linear(
            _moe_array((mixture_size, MODEL_DIM, HIDDEN_DIM), offset=offset + 200),
            jax.device_put(_moe_array((mixture_size, MODEL_DIM), offset=offset + 300), expert_bias_sharding),
            (MODEL_DIM,),
            is_sharded=is_sharded,
        ),
    )


def _expert_mlp_template(config: DenseMLPConfig, mixture_size: int, *, is_sharded: bool) -> DenseMLP:
    if config.has_up_biases:
        up_biases_shape = (mixture_size, 2 * HIDDEN_DIM)
    else:
        up_biases_shape = None
    if config.has_down_biases:
        down_biases_shape = (mixture_size, MODEL_DIM)
    else:
        down_biases_shape = None
    return DenseMLP(
        config=config,
        up_projection=_linear_template(
            (mixture_size, 2 * HIDDEN_DIM, MODEL_DIM),
            up_biases_shape,
            (HIDDEN_DIM, HIDDEN_DIM),
            (None, None),
            is_sharded=is_sharded,
        ),
        down_projection=_linear_template(
            (mixture_size, MODEL_DIM, HIDDEN_DIM),
            down_biases_shape,
            (MODEL_DIM,),
            (None, None),
            is_sharded=is_sharded,
        ),
    )


def _moe(
    *,
    num_active_routed_experts: int = 1,
    num_shared_experts: int = 0,
    has_gate: bool = False,
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
        gate_config=LinearConfig() if has_gate else None,
    )
    return MixtureOfExperts(
        config=config,
        router=_linear(
            _moe_array((NUM_ROUTED_EXPERTS, MODEL_DIM)),
            _moe_array((NUM_ROUTED_EXPERTS,), offset=100),
            (NUM_ROUTED_EXPERTS,),
            is_sharded=False,
        ),
        routed_experts=_expert_mlp(expert_config, NUM_ROUTED_EXPERTS, offset=200, is_sharded=True),
        shared_experts=_expert_mlp(expert_config, num_shared_experts, offset=700, is_sharded=False)
        if num_shared_experts > 0
        else None,
        gate=Linear(
            config=LinearConfig(),
            weights=FullPrecisionMatrix(
                spec=FullPrecisionSpec(),
                is_sharded=False,
                weights=_moe_array((1, MODEL_DIM), offset=600),
            ),
            biases=None,
            output_dims=(1,),
        )
        if has_gate
        else None,
    )


def _routed_only_moe(*, num_active_routed_experts: int = 1) -> MixtureOfExperts:
    return _moe(num_active_routed_experts=num_active_routed_experts)


def _moe_with_awq_experts(module: MixtureOfExperts) -> MixtureOfExperts:
    spec = AWQSpec(bits=8, group_size=2)
    up_projection = spec.compress(
        module.routed_experts.up_projection.weights.decompress(),
        implementation=CompressionImplementation.INFERENCE,
    )
    down_projection = spec.compress(
        module.routed_experts.down_projection.weights.decompress(),
        implementation=CompressionImplementation.INFERENCE,
    )
    return eqx.tree_at(
        lambda value: (
            value.routed_experts.up_projection.weights,
            value.routed_experts.down_projection.weights,
        ),
        module,
        (up_projection, down_projection),
    )


def _moe_template(module: MixtureOfExperts) -> MixtureOfExperts:
    if module.shared_experts is None:
        shared_experts = None
    else:
        shared_experts = _expert_mlp_template(
            module.shared_experts.config,
            module.config.num_shared_experts,
            is_sharded=False,
        )
    return MixtureOfExperts(
        config=module.config,
        router=_linear_template(
            (module.config.num_routed_experts, MODEL_DIM),
            (module.config.num_routed_experts,) if module.config.router_has_biases else None,
            (module.config.num_routed_experts,),
            (None,),
            is_sharded=False,
        ),
        routed_experts=_expert_mlp_template(
            module.routed_experts.config,
            module.config.num_routed_experts,
            is_sharded=True,
        ),
        shared_experts=shared_experts,
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


def _expert_reference(experts: DenseMLP, expert_idx: int, inputs: Array) -> Array:
    up_weights = jnp.asarray(jax.device_get(experts.up_projection.weights.decompress()))[expert_idx]
    down_weights = jnp.asarray(jax.device_get(experts.down_projection.weights.decompress()))[expert_idx]
    up_result = up_weights @ inputs
    if experts.up_projection.biases is not None:
        up_result = up_result + jnp.asarray(jax.device_get(experts.up_projection.biases))[expert_idx]
    up_projection, gate = jnp.split(up_result, (experts.hidden_dim,))
    hidden = up_projection * experts.config.activation(gate)
    result = down_weights @ hidden
    if experts.down_projection.biases is not None:
        result = result + jnp.asarray(jax.device_get(experts.down_projection.biases))[expert_idx]
    return result


def _shared_expert_weight_reference(module: MixtureOfExperts, inputs: Array) -> Array:
    if module.gate is None:
        return jnp.ones((), dtype=inputs.dtype)

    gate_weights = jnp.asarray(jax.device_get(module.gate.weights.decompress()))
    (gate_value,) = gate_weights @ inputs
    return jax.nn.sigmoid(gate_value)


def _routed_moe_token_reference(module: MixtureOfExperts, inputs: Array) -> Array:
    logits = _router_logits_reference(module, inputs)
    active_logits, active_indices = jax.lax.top_k(logits, module.config.num_active_routed_experts)
    active_weights = jax.nn.softmax(active_logits)
    result = jnp.zeros((module.model_dim,), dtype=inputs.dtype)
    for active_idx, active_weight in zip(jax.device_get(active_indices), active_weights, strict=True):
        result = result + _expert_reference(module.routed_experts, int(active_idx), inputs) * active_weight
    if module.shared_experts is not None:
        shared_weight = _shared_expert_weight_reference(module, inputs)
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


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_linear_template_restored(restored: Linear, template: Linear, fake_mesh: Mesh) -> None:
    assert isinstance(restored.weights, FullPrecisionMatrix)
    assert isinstance(template.weights, FullPrecisionMatrix)
    assert restored.weights.weights.sharding == template.weights.weights.sharding
    _assert_named_sharding(restored.weights.weights.sharding, fake_mesh)
    if restored.biases is None:
        assert template.biases is None
        return
    assert template.biases is not None
    assert restored.biases.sharding == template.biases.sharding


def _assert_mlp_template_restored(restored: DenseMLP, template: DenseMLP, fake_mesh: Mesh) -> None:
    _assert_linear_template_restored(restored.up_projection, template.up_projection, fake_mesh)
    _assert_linear_template_restored(restored.down_projection, template.down_projection, fake_mesh)


def _assert_linear_weight_sharding(
    linear: Linear,
    partition: tuple[ShardingAxis | None, ...],
    fake_mesh: Mesh,
) -> None:
    sharding = make_sharding(partition)
    assert sharding is not None
    assert isinstance(linear.weights, FullPrecisionMatrix)
    assert linear.weights.weights.sharding == sharding
    assert linear.weights.is_sharded is (ShardingAxis.EXPERT in partition)
    _assert_named_sharding(linear.weights.weights.sharding, fake_mesh)


def _sharded_vector(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None,)))


def _sharded_vectors(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA, None)))


def _sharded_sequences(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA, None, None)))


def _moe_sequence_length(mode: ForwardPassMode) -> int:
    if mode == ForwardPassMode.SINGLE_TOKEN:
        return 1
    return 3


def test_dense_mlp_call_unbatched_matches_reference_and_keeps_unsharded_features(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_vector(jnp.arange(MODEL_DIM, dtype=jnp.float32))

    result = module.call_unbatched(inputs, keychain=Keychain.init(0))

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_dense_mlp_call_unbatched_without_bias_or_clipping_matches_reference(fake_mesh: Mesh) -> None:
    module = _dense_mlp(
        DenseMLPConfig(
            linear_config=LinearConfig(),
            activation=Identity(),
            has_up_biases=False,
            has_down_biases=False,
            gate_clipping=None,
            up_clipping=None,
        ),
    )
    inputs = _sharded_vector(jnp.arange(MODEL_DIM, dtype=jnp.float32))

    result = module.call_unbatched(inputs, keychain=Keychain.init(1))

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


def test_dense_mlp_output_dtype_matches_input_dtype(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_vector(jnp.arange(MODEL_DIM, dtype=jnp.bfloat16))

    result = module.call_unbatched(inputs, keychain=Keychain.init(15))

    assert result.dtype == inputs.dtype
    _assert_named_sharding(result.sharding, fake_mesh)


def test_dense_mlp_under_jit_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_sequences(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)

    result = eqx.filter_jit(lambda module, values: module(values, keychain=Keychain.init(2)))(module, inputs)

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_dense_mlp_vmapped_over_inputs_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_vectors(jnp.arange(2 * MODEL_DIM, dtype=jnp.float32).reshape(2, MODEL_DIM) / 10)

    result = call_vmapped(
        module.call_unbatched,
        inputs,
        keychain=Keychain.init(3),
        added_sharding_axis=ShardingAxis.DATA,
    )

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))


def test_dense_mlp_full_call_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _dense_mlp()
    inputs = _sharded_sequences(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)

    result = module(
        inputs,
        keychain=Keychain.init(4),
    )

    _assert_close(result=result, reference=_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_dense_mlp_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _dense_mlp()
    weight_template = make_sharding((ShardingAxis.TENSOR, None))
    bias_template = make_sharding((None,))
    assert weight_template is not None
    assert bias_template is not None
    template = DenseMLP(
        config=original.config,
        up_projection=_linear(
            dummy_array(original.up_projection.weights.shape, jnp.float32, weight_template),
            dummy_array((2 * HIDDEN_DIM,), jnp.float32, bias_template),
            (HIDDEN_DIM, HIDDEN_DIM),
        ),
        down_projection=_linear(
            dummy_array(original.down_projection.weights.shape, jnp.float32, weight_template),
            dummy_array((MODEL_DIM,), jnp.float32, bias_template),
            (MODEL_DIM,),
        ),
    )
    inputs = _sharded_sequences(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)

    restored = template.load_exported(original.export())
    result = restored(inputs, keychain=Keychain.init(5))

    _assert_mlp_template_restored(restored, template, fake_mesh)
    _assert_close(result=result, reference=original(inputs, keychain=Keychain.init(6)))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_softmax_routing_call_unbatched_selects_top_k_and_normalizes_weights() -> None:
    routing = SoftmaxRouting()
    logits = jnp.array([1.0, 3.0, 2.0], dtype=jnp.float32)

    result = routing.call_unbatched(logits, num_active=2)

    assert jnp.array_equal(result.active_expert_indices, jnp.array([1, 2]))
    assert_close(result=result.active_expert_weights, reference=jax.nn.softmax(jnp.array([3.0, 2.0])))
    assert_close(result=jnp.sum(result.active_expert_weights), reference=jnp.array(1.0, dtype=jnp.float32))


def test_softmax_routing_vmapped_call_matches_unbatched() -> None:
    routing = SoftmaxRouting()
    logits = jnp.array(
        [
            [1.0, 3.0, 2.0],
            [4.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    result = routing(logits, num_active=2)
    reference = jax.vmap(lambda token_logits: routing.call_unbatched(token_logits, num_active=2))(logits)

    assert jnp.array_equal(result.active_expert_indices, reference.active_expert_indices)
    assert_close(result=result.active_expert_weights, reference=reference.active_expert_weights)


def test_moe_config_init_shards_routed_and_replicates_router_and_shared_experts(fake_mesh: Mesh) -> None:
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
        num_active_routed_experts=1,
        router_has_biases=False,
        num_shared_experts=1,
        expert_hidden_dim=HIDDEN_DIM,
    )
    module = config.init(
        RandomInitializer(dtype=jnp.float32, key=jax.random.key(21)),
        model_dim=MODEL_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    sharded_router_weights = FullPrecisionSpec().compress(module.router.weights.decompress())
    sharded_router_module = eqx.tree_at(lambda m: m.router.weights, module, sharded_router_weights)
    inputs = _sharded_sequences(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(27),
    )
    reference = sharded_router_module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(27),
    )

    assert module.shared_experts is not None
    _assert_linear_weight_sharding(module.router, (None, None), fake_mesh)
    for projection in (module.routed_experts.up_projection, module.routed_experts.down_projection):
        _assert_linear_weight_sharding(projection, (ShardingAxis.EXPERT, None, None), fake_mesh)
    for projection in (module.shared_experts.up_projection, module.shared_experts.down_projection):
        _assert_linear_weight_sharding(projection, (None, None, None), fake_mesh)
    _assert_close(result=result, reference=reference)


@pytest.mark.parametrize("mode", MOE_MODES)
def test_routed_moe_matches_direct_reference(fake_mesh: Mesh, mode: ForwardPassMode) -> None:
    assert fake_mesh is not None
    module = _routed_only_moe(num_active_routed_experts=2)
    sequence_length = _moe_sequence_length(mode)
    inputs = (
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
        keychain=Keychain.init(7),
    )

    _assert_close(result=result, reference=_routed_moe_reference(module, inputs))


def test_moe_prefill_with_compressed_experts_matches_direct_reference(fake_mesh: Mesh) -> None:
    assert fake_mesh is not None
    module = _moe_with_awq_experts(_routed_only_moe(num_active_routed_experts=2))
    inputs = jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(17),
    )

    _assert_close(result=result, reference=_routed_moe_reference(module, inputs))


@pytest.mark.parametrize("mode", MOE_MODES)
def test_routed_moe_output_dtype_matches_input_dtype(fake_mesh: Mesh, mode: ForwardPassMode) -> None:
    assert fake_mesh is not None
    module = _routed_only_moe(num_active_routed_experts=2)
    sequence_length = _moe_sequence_length(mode)
    inputs = (
        jnp.arange(2 * sequence_length * MODEL_DIM, dtype=jnp.bfloat16).reshape(
            2,
            sequence_length,
            MODEL_DIM,
        )
        / 10
    )

    result = module(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=mode),
        keychain=Keychain.init(16),
    )

    assert result.dtype == inputs.dtype


def test_moe_prefill_routed_only_with_padding_zeroes_padded_tokens(fake_mesh: Mesh) -> None:
    # Regression for the padded-token leak: padded (token, expert) pairs are routed to the sentinel
    # expert and dropped from the ragged_dot groups, but ragged_dot still emits nonzero rows for that
    # ungrouped tail. Their routing weights must be zeroed so padded positions stay exactly zero. This
    # bug lives in the routed path alone and does not require shared experts to reproduce.
    assert fake_mesh is not None
    module = _routed_only_moe(num_active_routed_experts=2)
    inputs = jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10
    lengths_without_padding = jnp.array([3, 1], dtype=jnp.int32)

    result = module(
        inputs,
        lengths_without_padding=lengths_without_padding,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(9),
    )

    _assert_close(
        result=result,
        reference=_routed_moe_reference(module, inputs, lengths_without_padding=lengths_without_padding),
    )
    # Padded positions (batch 1, tokens 1 and 2) must be exactly zero, not merely close.
    assert jnp.all(jnp.asarray(jax.device_get(result))[1, 1:] == 0)


def test_moe_prefill_with_shared_experts_and_padding_matches_direct_reference(fake_mesh: Mesh) -> None:
    assert fake_mesh is not None
    module = _moe(num_active_routed_experts=2, num_shared_experts=2)
    inputs = jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10
    lengths_without_padding = jnp.array([3, 1], dtype=jnp.int32)

    result = module(
        inputs,
        lengths_without_padding=lengths_without_padding,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(9),
    )

    _assert_close(
        result=result,
        reference=_routed_moe_reference(module, inputs, lengths_without_padding=lengths_without_padding),
    )


def test_moe_prefill_with_gated_shared_experts_matches_direct_reference(fake_mesh: Mesh) -> None:
    assert fake_mesh is not None
    module = _moe(num_active_routed_experts=2, num_shared_experts=2, has_gate=True)
    inputs = jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10
    lengths_without_padding = jnp.array([3, 1], dtype=jnp.int32)

    result = module(
        inputs,
        lengths_without_padding=lengths_without_padding,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(10),
    )

    _assert_close(
        result=result,
        reference=_routed_moe_reference(module, inputs, lengths_without_padding=lengths_without_padding),
    )


@pytest.mark.parametrize("mode", MOE_MODES)
def test_routed_moe_under_jit_matches_reference_and_keeps_data_sharding(
    fake_mesh: Mesh,
    mode: ForwardPassMode,
) -> None:
    module = _routed_only_moe(num_active_routed_experts=2)
    sequence_length = _moe_sequence_length(mode)
    inputs = _sharded_sequences(
        jnp.arange(2 * sequence_length * MODEL_DIM, dtype=jnp.float32).reshape(2, sequence_length, MODEL_DIM) / 10,
    )

    result = eqx.filter_jit(
        lambda module, inputs: module(
            inputs,
            forward_pass_config=MLPForwardPassConfig(mode=mode, moe_chunk_size_ratio=0.5),
            keychain=Keychain.init(11),
        )
    )(module, inputs)

    _assert_close(result=result, reference=_routed_moe_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None, None))


def test_moe_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _moe(num_active_routed_experts=2, num_shared_experts=2)
    template = _moe_template(original)
    inputs = _sharded_sequences(jnp.arange(2 * 3 * MODEL_DIM, dtype=jnp.float32).reshape(2, 3, MODEL_DIM) / 10)

    restored = template.load_exported(original.export())
    result = restored(
        inputs,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(13),
    )

    _assert_linear_template_restored(restored.router, template.router, fake_mesh)
    _assert_mlp_template_restored(restored.routed_experts, template.routed_experts, fake_mesh)
    assert restored.shared_experts is not None
    assert template.shared_experts is not None
    _assert_mlp_template_restored(restored.shared_experts, template.shared_experts, fake_mesh)
    _assert_close(
        result=result,
        reference=original(
            inputs,
            forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
            keychain=Keychain.init(14),
        ),
    )
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None, None))
