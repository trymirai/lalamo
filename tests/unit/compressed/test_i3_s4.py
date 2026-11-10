import jax
import jax.numpy as jnp
import numpy as np

from lalamo.compressed.i3_s4 import I3S4Matrix, I3S4Spec
from lalamo.initializer import EmptyInitializer
from lalamo.utils.sharding import LogicalAxis, ShardingConfig, sharding_of


def test_loaded_i3_s4_decompresses_with_row_sharding() -> None:
    config = ShardingConfig.fully_sharded_data_parallel(jax.devices("cpu"))
    generator = np.random.default_rng(0)
    with jax.set_mesh(config.mesh):
        source = I3S4Matrix(
            spec=I3S4Spec(),
            sharding_config=config,
            is_sharded=True,
            codes=jnp.asarray(generator.integers(256, size=(8, 48), dtype=np.uint8)),
            row_scales=jnp.linspace(0.5, 1, 8),
            ladder_indices=jnp.asarray(generator.integers(256, size=(8, 1), dtype=np.uint8)),
            ladder=jnp.linspace(0.5, 1, 16, dtype=jnp.float16),
            input_hadamard_factors=jnp.ones(128, dtype=jnp.int32),
        )
        reference = np.asarray(source.decompress())
        restored = EmptyInitializer(None, config).weight_matrix(*source.shape).load_exported(source.export())
        for weights in (restored.decompress(), restored.to_full_precision().decompress()):
            np.testing.assert_array_equal(weights, reference)
            assert sharding_of(weights).spec[0] == config.resolve_axis(LogicalAxis.MATRIX)
