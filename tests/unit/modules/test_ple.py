import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.activations import Identity
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.ple import PLELayer, PLELayerConfig
from lalamo.weight_matrix import FullPrecisionSpec
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config


def test_ple_layer_returns_projected_update_without_residual(fake_mesh: Mesh) -> None:  # noqa: ARG001
    sharding_config = make_test_sharding_config()
    linear_config = LinearConfig()
    module = PLELayer(
        config=PLELayerConfig(
            linear_config=linear_config,
            ple_channels=2,
            activation=Identity(),
        ),
        sharding_config=sharding_config,
        gate=Linear(
            config=linear_config,
            sharding_config=sharding_config,
            weights=FullPrecisionSpec().compress(
                jnp.eye(2, dtype=jnp.float32),
                sharding_config=sharding_config,
            ),
            biases=None,
            output_dims=(2,),
        ),
        projection=Linear(
            config=linear_config,
            sharding_config=sharding_config,
            weights=FullPrecisionSpec().compress(
                2 * jnp.eye(2, dtype=jnp.float32),
                sharding_config=sharding_config,
            ),
            biases=None,
            output_dims=(2,),
        ),
    )
    outputs = jax.device_put(
        jnp.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 3.0], [4.0, 5.0]],
            ],
        ),
        make_sharding((LogicalAxis.BATCH, None, None)),
    )
    per_layer_input = jax.device_put(
        jnp.array(
            [
                [[5.0, 6.0], [7.0, 8.0]],
                [[6.0, 7.0], [8.0, 9.0]],
            ],
        ),
        make_sharding((LogicalAxis.BATCH, None, None)),
    )

    projected_update = module(
        outputs,
        per_layer_input,
        keychain=Keychain.init(0, sharding_config=sharding_config),
    )

    assert_close(result=projected_update, reference=2 * outputs * per_layer_input)
