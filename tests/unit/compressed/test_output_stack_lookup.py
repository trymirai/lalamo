import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.hybrid import HybridMatrix
from lalamo.compressed.incoherence import HadamardSpec, IncoherenceProcessingMode
from lalamo.compressed.int import IntSpec
from lalamo.compressed.output_stack import OutputStackMatrix, OutputStackPart, OutputStackSpec
from lalamo.exportable import ExportResults
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import EmbeddingMatrix, Layout, ShapeDtypeSpec
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

MODEL_DIM = 128
PART_ROWS = (16, 48, 32)
ROWS = sum(PART_ROWS)


def stack(layout: Layout) -> OutputStackMatrix:
    # int8, int4 and int8 parts. Stored input-output, they are the rows of a table and stack along the vocabulary.
    sharding_config = make_test_sharding_config()
    specs = (
        IntSpec(8, 64, is_symmetric=True, layout=layout),
        IntSpec(4, 32, layout=layout),
        IntSpec(8, 64, is_symmetric=True, layout=layout),
    )
    parts = tuple(
        spec.compress(
            jax.random.normal(
                jax.random.key(index),
                (rows, MODEL_DIM) if layout == Layout.OUTPUT_INPUT else (MODEL_DIM, rows),
                jnp.float32,
            ),
            sharding_config=sharding_config,
        )
        for index, (spec, rows) in enumerate(zip(specs, PART_ROWS, strict=True))
    )
    spec = OutputStackSpec(
        tuple(OutputStackPart(rows, part.spec) for rows, part in zip(PART_ROWS, parts, strict=True))
    )
    return OutputStackMatrix(spec, sharding_config, parts[0].is_sharded, parts)


def test_table_stack_looks_every_row_up_in_the_part_that_holds_it() -> None:
    table = stack(Layout.INPUT_OUTPUT)
    keychain = Keychain.init(0, sharding_config=table.sharding_config)
    order = np.random.default_rng(0).permutation(ROWS)
    found = np.asarray(table.lookup_embedding(jnp.asarray(order, jnp.int32), keychain=keychain))

    start = 0
    for part, rows in zip(table.parts, PART_ROWS, strict=True):
        assert isinstance(part, EmbeddingMatrix)
        own = np.asarray(part.lookup_embedding(jnp.arange(rows, dtype=jnp.int32), keychain=keychain))
        inside = (order >= start) & (order < start + rows)
        np.testing.assert_array_equal(found[inside], own[order[inside] - start])
        start += rows
    np.testing.assert_array_equal(np.asarray(table.lookup_embedding(17, keychain=keychain)), found[order == 17][0])


def test_table_stack_decodes_to_its_rows_along_the_vocabulary() -> None:
    table = stack(Layout.INPUT_OUTPUT)
    keychain = Keychain.init(0, sharding_config=table.sharding_config)
    rows = np.asarray(table.lookup_embedding(jnp.arange(ROWS, dtype=jnp.int32), keychain=keychain))

    assert table.shape == (ROWS, MODEL_DIM)
    assert table.spec.layout == Layout.INPUT_OUTPUT
    np.testing.assert_array_equal(np.asarray(table.decompress()), rows.T)
    dense = table.to_full_precision()
    assert dense.spec.layout == Layout.INPUT_OUTPUT
    np.testing.assert_array_equal(np.asarray(dense.decompress()), rows.T)


def test_table_stack_survives_export_and_load_inside_a_hybrid() -> None:
    table = stack(Layout.INPUT_OUTPUT)
    sharding_config = table.sharding_config
    rotation = HadamardSpec(32, IncoherenceProcessingMode.OUTPUT).init(
        ROWS, MODEL_DIM, jax.random.key(7), sharding_config
    )
    rotated = HybridMatrix.of(table, rotation)
    arrays, metadata = rotated.export()
    template = ShapeDtypeSpec(Layout.INPUT_OUTPUT).compress(
        dummy_array((MODEL_DIM, ROWS), jnp.float32, sharding_config.resolve_sharding((None, None))),
        sharding_config=sharding_config,
    )

    loaded = template.load_exported(ExportResults(arrays, {**metadata, ParameterPath("spec"): rotated.spec.to_json()}))

    assert isinstance(loaded, HybridMatrix)
    assert isinstance(loaded.quantized, OutputStackMatrix)
    assert [part.shape for part in loaded.quantized.parts] == [(rows, MODEL_DIM) for rows in PART_ROWS]
    keychain = Keychain.init(0, sharding_config=sharding_config)
    rows = jnp.arange(ROWS, dtype=jnp.int32)
    looked_up = np.asarray(rotated.lookup_embedding(rows, keychain=keychain))
    np.testing.assert_array_equal(np.asarray(loaded.lookup_embedding(rows, keychain=keychain)), looked_up)
    # The rotation is applied once, to the rows read from their parts.
    unrotated = table.lookup_embedding(rows, keychain=keychain, dtype=jnp.float32)
    np.testing.assert_array_equal(looked_up, np.asarray(rotation.unprocess_embedding(unrotated)))


def test_stack_of_output_rows_stays_a_projection() -> None:
    projection = stack(Layout.OUTPUT_INPUT)
    keychain = Keychain.init(0, sharding_config=projection.sharding_config)
    vector = jax.random.normal(jax.random.key(3), (MODEL_DIM,), jnp.float32)

    assert projection.spec.layout == Layout.OUTPUT_INPUT
    assert projection.shape == (ROWS, MODEL_DIM)
    outputs = [np.asarray(part.dot(vector, keychain=keychain)) for part in projection.parts]
    np.testing.assert_array_equal(np.asarray(projection.dot(vector, keychain=keychain)), np.concatenate(outputs))
    decoded = [np.asarray(part.decompress()) for part in projection.parts]
    np.testing.assert_array_equal(np.asarray(projection.decompress()), np.concatenate(decoded))
    assert projection.to_full_precision().spec.layout == Layout.OUTPUT_INPUT
    with pytest.raises(ValueError, match="input-output layout"):
        projection.lookup_embedding(0, keychain=keychain)
