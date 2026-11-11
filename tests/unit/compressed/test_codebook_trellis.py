from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.codebook_trellis import CodebookTrellisMatrix, CodebookTrellisSpec
from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.safetensors import safe_read, safe_write
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import Layout

pytestmark = pytest.mark.usefixtures("fake_mesh")


def test_custom_table_survives_serialization_and_sharded_inference(tmp_path: Path) -> None:
    config = ShardingConfig.fully_sharded_data_parallel(jax.devices("cpu"))
    with jax.set_mesh(config.mesh):
        generator = np.random.default_rng(42)
        table = generator.normal(size=(65536, 2)).astype(np.float32)
        source = CodebookTrellisMatrix(
            spec=CodebookTrellisSpec(2, 4, 0),
            sharding_config=config,
            is_sharded=True,
            dtype_=jnp.float32,
            columns=128,
            codes=jnp.asarray(generator.integers(256, size=(8, 34), dtype=np.uint8)),
            scales=jnp.linspace(0.5, 1, 8),
            table=jnp.asarray(table),
        )
        reference = table[np.asarray(source.spec.states(source.codes, 128))].reshape(8, 128)
        reference *= np.asarray(source.scales)[:, None]
        exported = source.export()
        path = tmp_path / "weights.safetensors"
        with path.open("wb") as stream:
            safe_write(stream, exported.arrays)
        with path.open("rb") as stream:
            _, arrays, _ = safe_read(stream)
            restored = (
                EmptyInitializer(None, config)
                .weight_matrix(8, 128)
                .load_exported(ExportResults(arrays, exported.metadata))
            )
        assert isinstance(restored, CodebookTrellisMatrix)
        np.testing.assert_array_equal(restored.table, table)
        np.testing.assert_array_equal(restored.decompress(), reference)
        keychain = Keychain.init(0, sharding_config=config)
        vector = jnp.linspace(-1, 1, 128)
        np.testing.assert_allclose(
            np.asarray(restored.dot(vector, keychain=keychain)), reference @ np.asarray(vector), atol=1e-4
        )
        embedding = replace(restored, spec=replace(restored.spec, layout=Layout.INPUT_OUTPUT))
        indices = jnp.arange(8, dtype=jnp.int32)
        np.testing.assert_array_equal(embedding.lookup_embedding(indices, keychain=keychain), reference)
