from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from lalamo.initializer import RandomInitializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.linear import LinearConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConv, SeparableCausalConvConfig
from lalamo.modules.transformer_layer import TransformerLayerConvConfig
from lalamo.weight_matrix import MatmulConfig
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder, make_test_sharding_config


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


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize("group_size", [1, 2])
def test_dflash_sublayer_transform_matches_explicit_reference(group_size: int) -> None:
    sharding_config = make_test_sharding_config()
    config = TransformerLayerConvConfig(
        conv_config=SeparableCausalConvConfig(has_biases=False),
        kernel_projection_config=LinearConfig(),
        conv_kernel_size=2,
        conv_group_size=group_size,
    )
    module = config.init(
        RandomInitializer(jnp.float32, sharding_config, key=jax.random.key(0)),
        model_dim=4,
    )
    inputs = jax.device_put(
        jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4) / 10,
        sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None)),
    )
    keychain = Keychain.init(0, sharding_config=sharding_config)

    prepared, finishing_coefficients = module.prepare(
        inputs,
        MatmulConfig(),
        keychain=keychain,
    )

    projection_weights = module.kernel_projection.weights.decompress()
    projected = jnp.einsum("btc,oc->bto", inputs, projection_weights)
    coefficients = projected.reshape(2, 4, 2, module.pre_conv.kernel_size, 4 // group_size)
    reference_prepared = reference_convolution(module.pre_conv, inputs, coefficients[:, :, 0])
    reference_finished = reference_convolution(module.post_conv, reference_prepared, coefficients[:, :, 1])

    assert_close(result=prepared, reference=reference_prepared)
    assert_close(
        result=module.finish(prepared, finishing_coefficients),
        reference=reference_finished,
    )


@pytest.mark.parametrize(("mixer_conv_enabled", "mlp_conv_enabled"), [(True, False), (False, True), (True, True)])
def test_sublayer_transform_preserves_suffix(mixer_conv_enabled: bool, mlp_conv_enabled: bool) -> None:
    decoder = build_tiny_attention_decoder((None,))
    conv_config = TransformerLayerConvConfig(
        conv_config=SeparableCausalConvConfig(has_biases=False),
        kernel_projection_config=LinearConfig(),
        conv_kernel_size=2,
        conv_group_size=2,
    )
    mixer_conv_config = None
    mlp_conv_config = None
    if mixer_conv_enabled:
        mixer_conv_config = conv_config
    if mlp_conv_enabled:
        mlp_conv_config = conv_config
    config = replace(
        decoder.transformer.layers[0].config,
        mixer_conv_config=mixer_conv_config,
        mlp_conv_config=mlp_conv_config,
    )
    layer = config.init(
        RandomInitializer(jnp.float32, decoder.sharding_config, key=jax.random.key(0)),
        model_dim=8,
        hidden_dim=16,
    )
    assert (layer.mixer_conv is not None) == mixer_conv_enabled
    assert (layer.mlp_conv is not None) == mlp_conv_enabled
    inputs = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 10
    keychain = Keychain.init(0, sharding_config=decoder.sharding_config)

    full_result = layer(inputs, None, keychain=keychain)
    suffix_result = layer(inputs, None, return_suffix_tokens=1, keychain=keychain)

    assert_close(result=suffix_result.outputs, reference=full_result.outputs[:, -1:])
