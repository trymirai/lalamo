import jax
import jax.numpy as jnp

from lalamo.module import Keychain
from lalamo.modules import DFlashGroupedConvolution, DFlashGroupedConvolutionConfig, Linear, LinearConfig
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import FullPrecisionSpec, MatmulConfig
from tests.common import assert_close


def make_grouped_convolution() -> DFlashGroupedConvolution:
    sharding_config = ShardingConfig.replicated(jax.devices("cpu")[:1])
    kernel_size = 2
    group_size = 2
    model_dim = 4
    num_groups = model_dim // group_size
    projection_output_dim = 2 * kernel_size * num_groups
    projection_weights = (
        jnp.arange(projection_output_dim * model_dim, dtype=jnp.float32).reshape(
            projection_output_dim,
            model_dim,
        )
        / 100
    )
    return DFlashGroupedConvolution(
        config=DFlashGroupedConvolutionConfig(
            kernel_size=kernel_size,
            group_size=group_size,
            kernel_projection_config=LinearConfig(),
        ),
        sharding_config=sharding_config,
        base_kernel=jnp.arange(2 * kernel_size * model_dim, dtype=jnp.float32).reshape(
            2,
            kernel_size,
            model_dim,
        )
        / 10,
        kernel_projection=Linear(
            config=LinearConfig(),
            sharding_config=sharding_config,
            weights=FullPrecisionSpec().compress(
                projection_weights,
                sharding_config=sharding_config,
                is_sharded=False,
            ),
            biases=None,
            output_dims=(projection_output_dim,),
        ),
    )


def reference_convolution(
    module: DFlashGroupedConvolution,
    hidden_states: jax.Array,
    coefficient_deltas: jax.Array,
    side: int,
) -> jax.Array:
    batch_size, block_size, model_dim = hidden_states.shape
    hidden_groups = hidden_states.reshape(batch_size, block_size, module.num_groups, module.config.group_size)
    base_kernel = module.base_kernel[side].reshape(
        module.config.kernel_size,
        module.num_groups,
        module.config.group_size,
    )
    output_rows = []
    for token_index in range(block_size):
        token_output = jnp.zeros_like(hidden_groups[:, token_index])
        for tap in range(module.config.kernel_size):
            if token_index < tap:
                continue
            coefficients = base_kernel[tap][None] + coefficient_deltas[:, token_index, tap, :, None]
            token_output = token_output + coefficients * hidden_groups[:, token_index - tap]
        output_rows.append(token_output)
    return jnp.stack(output_rows, axis=1).reshape(batch_size, block_size, model_dim)


def test_dflash_grouped_convolution_matches_explicit_reference() -> None:
    module = make_grouped_convolution()
    inputs = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4) / 10
    keychain = Keychain.init(0, sharding_config=module.sharding_config)

    prepared, finishing_coefficients = module.prepare(inputs, MatmulConfig(), keychain=keychain)

    projection_weights = module.kernel_projection.weights.decompress()
    projected = jnp.einsum("btc,oc->bto", inputs, projection_weights)
    coefficients = projected.reshape(2, 4, 2, module.config.kernel_size, module.num_groups)
    reference_prepared = reference_convolution(module, inputs, coefficients[:, :, 0], side=0)
    reference_finished = reference_convolution(module, reference_prepared, coefficients[:, :, 1], side=1)

    assert_close(result=prepared, reference=reference_prepared)
    assert_close(result=module.finish(prepared, finishing_coefficients), reference=reference_finished)
