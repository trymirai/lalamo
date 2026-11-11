from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import hadamard

from lalamo.compressed.direction import DirectionMatrix, DirectionSpec
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import Layout

pytestmark = pytest.mark.usefixtures("fake_mesh")


@pytest.mark.parametrize("layout", list(Layout))
def test_dense_reference_reload_lookup_and_dot(layout: Layout) -> None:
    config = ShardingConfig.replicated(jax.devices("cpu")[:1])
    with jax.set_mesh(config.mesh):
        generator = np.random.default_rng(42)
        indices = generator.integers(8, size=(8, 2048), dtype=np.uint8)
        bits = ((indices[..., None] >> np.arange(3)) & 1).reshape(8, -1)
        codes = np.packbits(bits, axis=-1, bitorder="little")
        levels = np.linspace(-1.7, 1.7, 8, dtype=np.float32)
        tail = jnp.asarray(generator.normal(size=(8, 32)), dtype=jnp.bfloat16)
        matrix = DirectionMatrix(
            spec=DirectionSpec(layout),
            sharding_config=config,
            is_sharded=True,
            codes=jnp.asarray(codes),
            levels=jnp.asarray(levels),
            unit_scale=jnp.asarray(0.5, dtype=jnp.float32),
            mean_norm=jnp.asarray(1.25, dtype=jnp.float32),
            tail=tail,
        )
        values = levels[indices].astype(np.float64) * 0.5
        values *= 1.25 / np.linalg.norm(values, axis=-1, keepdims=True)
        rotated = (values.reshape(8, 2, 1024) @ (hadamard(1024) / 32)).reshape(8, 2048)
        rounded = np.asarray(jnp.asarray(rotated.astype(np.float32), dtype=jnp.bfloat16), dtype=np.float32)
        expected = np.concatenate((rounded, np.asarray(tail, dtype=np.float32)), axis=-1)
        transposed = layout == Layout.INPUT_OUTPUT
        initializer = EmptyInitializer(None, config)
        if transposed:
            template = initializer.embedding_matrix(*matrix.shape)
        else:
            template = initializer.weight_matrix(*matrix.shape)
        restored = template.load_exported(matrix.export())
        assert isinstance(restored, DirectionMatrix)
        decode = jax.jit(lambda m: m.decompress())
        actual = decode(restored)
        # Dense FP64 sums and butterflies leave different tiny residuals at exact cancellation zeros.
        np.testing.assert_allclose(actual, expected.T if transposed else expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(actual, decode(matrix))
        np.testing.assert_allclose(
            np.asarray(decode(restored.astype(jnp.float32))), np.asarray(actual, dtype=np.float32), rtol=0, atol=1e-15
        )
        keychain = Keychain.init(0, sharding_config=config)
        if transposed:
            indices = jnp.array([5, 0, 3], dtype=jnp.int32)
            np.testing.assert_allclose(
                np.asarray(restored.lookup_embedding(indices, keychain=keychain)),
                expected[[5, 0, 3]],
                rtol=0,
                atol=1e-15,
            )
        vector = jnp.linspace(-1, 1, expected.shape[1], dtype=jnp.float32)
        # More than 128 rows exercises both the full decode batch and its remainder.
        restored = replace(restored, codes=jnp.tile(restored.codes, (20, 1)), tail=jnp.tile(restored.tail, (20, 1)))
        result = jax.jit(lambda m: m.dot(vector, transposed=transposed, keychain=keychain))(restored)
        np.testing.assert_allclose(result, np.tile(expected, (20, 1)) @ np.asarray(vector), atol=1e-4, rtol=2e-5)
