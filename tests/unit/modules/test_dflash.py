import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer, RandomInitializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.flow import (
    Context,
    EmitConfig,
    FlowForwardPassConfig,
    SequenceConfig,
    SinkConfig,
    StepConfig,
    leaves,
    residual,
    run,
)
from lalamo.modules.linear import LinearConfig
from lalamo.modules.speculators.dflash import DFlashDraftConfig
from lalamo.modules.token_mixers.convolutions import (
    DynamicConv,
    DynamicConvConfig,
    DynamicConvKernels,
    DynamicConvKernelsConfig,
    SeparableCausalConv,
    SeparableCausalConvConfig,
)
from lalamo.safetensors import safe_read, safe_write
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close
from tests.helpers import make_test_sharding_config, tiny_attention_config, tiny_mlp_config, tiny_norm_config


def reference_convolution(
    convolution: SeparableCausalConv,
    hidden_states: jax.Array,
    coefficient_deltas: jax.Array,
) -> jax.Array:
    batch_size, block_size, model_dim = hidden_states.shape
    group_size = model_dim // coefficient_deltas.shape[-1]
    num_groups = model_dim // group_size
    hidden_groups = hidden_states.reshape(batch_size, block_size, num_groups, group_size)
    output_rows = []
    for token_index in range(block_size):
        token_output = jnp.zeros_like(hidden_groups[:, token_index])
        for tap in range(convolution.kernel_size):
            if token_index < tap:
                continue
            base_kernel = convolution.weights[:, convolution.kernel_size - 1 - tap].reshape(num_groups, group_size)
            coefficients = base_kernel[None] + coefficient_deltas[:, token_index, tap, :, None]
            token_output = token_output + coefficients * hidden_groups[:, token_index - tap]
        output_rows.append(token_output)
    return jnp.stack(output_rows, axis=1).reshape(batch_size, block_size, model_dim)


def convolved(*body: StepConfig, group_size: int = 2) -> tuple[StepConfig, ...]:
    kernels = EmitConfig(
        "conv_kernels", DynamicConvKernelsConfig(LinearConfig(), kernel_size=2, group_size=group_size)
    )
    conv_config = SeparableCausalConvConfig(has_biases=False)
    return (
        kernels,
        SinkConfig((kernels,), DynamicConvConfig(conv_config, kernel_size=2, side=0)),
        *body,
        SinkConfig((kernels,), DynamicConvConfig(conv_config, kernel_size=2, side=1)),
    )


def context(token_positions: jax.Array, sharding_config: ShardingConfig) -> Context:
    return Context(None, token_positions, None, None, FlowForwardPassConfig.for_tracer_tests(), sharding_config)


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize("group_size", [1, 2])
def test_dynamic_convolutions_match_explicit_reference(group_size: int) -> None:
    sharding_config = make_test_sharding_config()
    program = SequenceConfig(convolved(group_size=group_size)).init(
        RandomInitializer(jnp.float32, sharding_config, key=jax.random.key(0)),
        model_dim=4,
    )
    kernels, pre_conv, post_conv = leaves(program)
    assert isinstance(kernels, DynamicConvKernels)
    assert isinstance(pre_conv, DynamicConv)
    assert isinstance(post_conv, DynamicConv)
    inputs = jax.device_put(
        jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4) / 10,
        sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None)),
    )
    positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (2, 4))

    result = run(
        program,
        inputs,
        context(positions, sharding_config),
        keychain=Keychain.init(0, sharding_config=sharding_config),
    )

    projection_weights = kernels.projection.weights.decompress()
    projected = jnp.einsum("btc,oc->bto", inputs, projection_weights)
    coefficients = projected.reshape(2, 4, 2, pre_conv.conv.kernel_size, 4 // group_size)
    reference_prepared = reference_convolution(pre_conv.conv, inputs, coefficients[:, :, 0])
    reference_finished = reference_convolution(post_conv.conv, reference_prepared, coefficients[:, :, 1])
    assert_close(result=result.outputs, reference=reference_finished)


@pytest.mark.parametrize(("mixer_conv_enabled", "mlp_conv_enabled"), [(True, False), (False, True), (True, True)])
def test_dynamic_convolutions_preserve_suffix(mixer_conv_enabled: bool, mlp_conv_enabled: bool) -> None:
    mixer: tuple[StepConfig, ...] = (tiny_attention_config(),)
    mlp: tuple[StepConfig, ...] = (tiny_mlp_config(),)
    if mixer_conv_enabled:
        mixer = convolved(*mixer)
    if mlp_conv_enabled:
        mlp = convolved(*mlp)
    sharding_config = ShardingConfig.replicated(jax.devices("cpu")[:8])
    program = SequenceConfig(
        (residual(tiny_norm_config(), *mixer), residual(tiny_norm_config(), *mlp)),
    ).init(RandomInitializer(jnp.float32, sharding_config, key=jax.random.key(0)), model_dim=8)
    inputs = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 10
    ctx = context(jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (2, 4)), sharding_config)
    keychain = Keychain.init(0, sharding_config=sharding_config)

    full_result = run(program, inputs, ctx, keychain=keychain)
    suffix_result = run(program, inputs, ctx, keychain=keychain, suffix=1)

    assert_close(result=suffix_result.outputs, reference=full_result.outputs[:, -1:])


def test_dflash_forward_chunked_append_and_native_roundtrip(tmp_path: Path) -> None:
    layers = tuple(
        step
        for _ in range(2)
        for step in (
            residual(tiny_norm_config(), replace(tiny_attention_config(), is_causal=False)),
            residual(tiny_norm_config(), tiny_mlp_config()),
        )
    )
    config = DFlashDraftConfig(
        model_dim=8,
        block_size=3,
        mask_token_id=15,
        target_layer_ids=(0,),
        num_target_layers=1,
        vocab_size=32,
        context_projection_config=LinearConfig(),
        context_norm_config=tiny_norm_config(),
        program=SequenceConfig((*layers, tiny_norm_config())),
    )
    sharding = ShardingConfig.replicated(jax.devices("cpu")[:8])
    keychain = Keychain.init(0, sharding_config=sharding)
    forward = FlowForwardPassConfig.for_tracer_tests()
    with jax.set_mesh(sharding.mesh):
        model = config.init(RandomInitializer(jnp.float32, sharding, key=jax.random.key(7)))
        assert len(model.attentions) == 2
        model = replace(model, state_kv_projection=model.state_kv_projection_from_attentions())
        features = jnp.arange(64, dtype=jnp.float32).reshape(2, 4, 8) / 64
        positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (2, 4))
        lengths = jnp.array([4, 3], dtype=jnp.int32)
        initial = model.empty_state(2, 8, jnp.float32)
        full = model.append_state(initial, features, positions, lengths, forward, keychain=keychain)
        chunked = initial
        for start, end in ((0, 2), (2, 4)):
            chunked = model.append_state(
                chunked,
                features[:, start:end],
                positions[:, start:end],
                jnp.clip(lengths - start, 0, end - start),
                forward,
                keychain=keychain,
            )
        noise = jnp.linspace(-1, 1, 48, dtype=jnp.float32).reshape(2, 3, 8)
        expected = model(noise, full, lengths - 1, forward, keychain=keychain)
        assert_close(result=model(noise, chunked, lengths - 1, forward, keychain=keychain), reference=expected)
        exported = model.export()
        with (tmp_path / "draft.safetensors").open("wb") as stream:
            safe_write(stream, exported.arrays, {name: json.dumps(value) for name, value in exported.metadata.items()})
        restored = DFlashDraftConfig.from_json(config.to_json()).init(EmptyInitializer(jnp.float32, sharding))
        with (tmp_path / "draft.safetensors").open("rb") as stream:
            metadata, arrays, _ = safe_read(stream)
            assert metadata is not None
            restored = restored.load_exported(ExportResults(arrays, {k: json.loads(v) for k, v in metadata.items()}))
        assert_close(result=restored(noise, full, lengths - 1, forward, keychain=keychain), reference=expected)
