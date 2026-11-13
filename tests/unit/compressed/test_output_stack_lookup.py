import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.hybrid import HybridMatrix, HybridSpec
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
PART_ROWS = (16, 48)
ROWS = sum(PART_ROWS)


def table_stack() -> OutputStackMatrix:
    # An int8 part over an int4 part, stored input-output: the stack runs along the vocabulary.
    sharding_config = make_test_sharding_config()
    layout = Layout.INPUT_OUTPUT
    specs = (IntSpec(8, 64, is_symmetric=True, layout=layout), IntSpec(4, 32, layout=layout))
    parts = tuple(
        spec.compress(
            jax.random.normal(jax.random.key(index), (MODEL_DIM, rows), jnp.float32),
            sharding_config=sharding_config,
        )
        for index, (spec, rows) in enumerate(zip(specs, PART_ROWS, strict=True))
    )
    spec = OutputStackSpec(
        tuple(OutputStackPart(rows, part.spec) for rows, part in zip(PART_ROWS, parts, strict=True))
    )
    return OutputStackMatrix(spec, sharding_config, parts[0].is_sharded, parts)


def test_table_stack_looks_every_row_up_in_the_part_that_holds_it() -> None:
    stack = table_stack()
    keychain = Keychain.init(0, sharding_config=stack.sharding_config)
    order = np.random.default_rng(0).permutation(ROWS)
    found = np.asarray(stack.lookup_embedding(jnp.asarray(order, jnp.int32), keychain=keychain))

    start = 0
    for part, rows in zip(stack.parts, PART_ROWS, strict=True):
        assert isinstance(part, EmbeddingMatrix)
        own = np.asarray(part.lookup_embedding(jnp.arange(rows, dtype=jnp.int32), keychain=keychain))
        inside = (order >= start) & (order < start + rows)
        np.testing.assert_array_equal(found[inside], own[order[inside] - start])
        start += rows
    np.testing.assert_array_equal(np.asarray(stack.lookup_embedding(17, keychain=keychain)), found[order == 17][0])


def test_table_stack_decodes_to_its_rows_along_the_vocabulary() -> None:
    stack = table_stack()
    keychain = Keychain.init(0, sharding_config=stack.sharding_config)
    rows = np.asarray(stack.lookup_embedding(jnp.arange(ROWS, dtype=jnp.int32), keychain=keychain))

    assert stack.shape == (ROWS, MODEL_DIM)
    assert stack.spec.layout == Layout.INPUT_OUTPUT
    np.testing.assert_array_equal(np.asarray(stack.decompress()), rows.T)
    dense = stack.to_full_precision()
    assert dense.spec.layout == Layout.INPUT_OUTPUT
    np.testing.assert_array_equal(np.asarray(dense.decompress()), rows.T)


def test_table_stack_survives_export_and_load_inside_a_hybrid() -> None:
    stack = table_stack()
    sharding_config = stack.sharding_config
    rotation = HadamardSpec(32, IncoherenceProcessingMode.OUTPUT).init(
        ROWS, MODEL_DIM, jax.random.key(7), sharding_config
    )
    table = HybridMatrix.of(stack, rotation)
    arrays, metadata = table.export()
    template = ShapeDtypeSpec(Layout.INPUT_OUTPUT).compress(
        dummy_array((MODEL_DIM, ROWS), jnp.float32, sharding_config.resolve_sharding((None, None))),
        sharding_config=sharding_config,
    )

    loaded = template.load_exported(ExportResults(arrays, {**metadata, ParameterPath("spec"): table.spec.to_json()}))

    assert isinstance(loaded, HybridMatrix)
    assert isinstance(loaded.quantized, OutputStackMatrix)
    assert [part.shape for part in loaded.quantized.parts] == [(rows, MODEL_DIM) for rows in PART_ROWS]
    keychain = Keychain.init(0, sharding_config=sharding_config)
    rows = jnp.arange(ROWS, dtype=jnp.int32)
    looked_up = np.asarray(table.lookup_embedding(rows, keychain=keychain))
    np.testing.assert_array_equal(np.asarray(loaded.lookup_embedding(rows, keychain=keychain)), looked_up)
    # The rotation is applied once, to the rows read from their parts.
    unrotated = stack.lookup_embedding(rows, keychain=keychain, dtype=jnp.float32)
    np.testing.assert_array_equal(looked_up, np.asarray(rotation.unprocess_embedding(unrotated)))


def test_table_stack_of_rotated_parts_rotates_each_row_in_its_part() -> None:
    # Every part is a hybrid with a rotation of its own block size, as in a readout stack.
    sharding_config = make_test_sharding_config()
    layout = Layout.INPUT_OUTPUT
    plans = ((IntSpec(8, 64, is_symmetric=True, layout=layout), 64), (IntSpec(4, 32, layout=layout), 32))
    parts = tuple(
        HybridSpec(spec, None, HadamardSpec(block, IncoherenceProcessingMode.OUTPUT)).compress(
            jax.random.normal(jax.random.key(index), (MODEL_DIM, rows), jnp.float32),
            key=jax.random.key(index),
            sharding_config=sharding_config,
        )
        for index, ((spec, block), rows) in enumerate(zip(plans, PART_ROWS, strict=True))
    )
    spec = OutputStackSpec(
        tuple(OutputStackPart(rows, part.spec) for rows, part in zip(PART_ROWS, parts, strict=True))
    )
    stack = OutputStackMatrix(spec, sharding_config, parts[0].is_sharded, parts)
    keychain = Keychain.init(0, sharding_config=sharding_config)

    assert spec.layout == Layout.INPUT_OUTPUT
    found = np.asarray(stack.lookup_embedding(jnp.arange(ROWS, dtype=jnp.int32), keychain=keychain))
    own = [
        np.asarray(part.lookup_embedding(jnp.arange(rows), keychain=keychain))
        for part, rows in zip(parts, PART_ROWS, strict=True)
    ]
    np.testing.assert_array_equal(found, np.concatenate(own))
    np.testing.assert_array_equal(np.asarray(stack.decompress()), found.T)


def test_stack_rejects_parts_of_different_layouts() -> None:
    parts = (
        OutputStackPart(16, IntSpec(4, 32, layout=Layout.OUTPUT_INPUT)),
        OutputStackPart(48, IntSpec(4, 32, layout=Layout.INPUT_OUTPUT)),
    )
    with pytest.raises(ValueError, match="one layout"):
        OutputStackSpec(parts)
