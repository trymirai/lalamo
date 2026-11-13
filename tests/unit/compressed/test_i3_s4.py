from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed import HybridMatrix, I3S4Matrix, I3S4Spec
from lalamo.compressed.incoherence import HadamardProcessor
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.utils.sharding import LogicalAxis, ShardingConfig, sharding_of


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_loaded_i3_s4_matches_dense_reference(block_size: Literal[32, 64, 128]) -> None:
    config = ShardingConfig.fully_sharded_data_parallel(jax.devices("cpu"))
    generator = np.random.default_rng(0)
    with jax.set_mesh(config.mesh):
        quantized = I3S4Matrix(
            spec=I3S4Spec(),
            sharding_config=config,
            is_sharded=True,
            codes=jnp.asarray(generator.integers(256, size=(8, 48), dtype=np.uint8)),
            row_scales=jnp.linspace(0.5, 1, 8),
            ladder_indices=jnp.asarray(generator.integers(256, size=(8, 1), dtype=np.uint8)),
            ladder=jnp.linspace(0.5, 1, 16, dtype=jnp.float16),
        )
        input_signs = jnp.asarray(generator.choice(np.array([-1, 1], dtype=np.int32), size=128))
        source = HybridMatrix.of(quantized, HadamardProcessor.of(input_signs, None, block_size=block_size))
        bits = np.unpackbits(np.asarray(quantized.codes), axis=-1, bitorder="little").reshape(8, 128, 3)
        levels = 2 * np.sum(bits * np.array([1, 2, 4], dtype=np.float32), axis=-1) - 7
        indices = np.asarray(quantized.ladder_indices)
        groups = np.stack((indices & 15, indices >> 4), axis=-1).reshape(8, 2)
        scales = np.asarray(quantized.row_scales)[:, None] * np.asarray(quantized.ladder)[groups]
        rotated = (levels.reshape(8, 2, 64) * scales[..., None]).reshape(8, 128)
        hadamard = np.ones((1, 1), dtype=np.float32)
        for _ in range(block_size.bit_length() - 1):
            hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
        reference = (rotated.reshape(8, -1, block_size) @ hadamard / np.sqrt(np.float32(block_size))).reshape(8, 128)
        reference *= np.asarray(input_signs)
        rows, columns = source.shape
        restored = EmptyInitializer(None, config).weight_matrix(rows, columns).load_exported(source.export())
        for weights in (restored.decompress(), restored.to_full_precision().decompress()):
            np.testing.assert_allclose(weights, reference, atol=3e-6, rtol=1e-6)
            assert sharding_of(weights).spec[0] == config.resolve_axis(LogicalAxis.MATRIX)
        vector = jax.device_put(generator.normal(size=128).astype(np.float32), config.make_sharding((None,)))
        result = restored.dot(vector, keychain=Keychain.init(0, sharding_config=config))
        np.testing.assert_allclose(result, reference @ np.asarray(vector), atol=3e-5, rtol=1e-5)
