from math import prod

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import (
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
)
from tests.common import assert_close_arrays, assert_named_sharding
from tests.helpers import make_sharding, make_test_sharding_config


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 4)
    return jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) / 10


def _stored_weights(layout: Layout, weights: jax.Array) -> jax.Array:
    if layout == Layout.INPUT_OUTPUT:
        return weights.swapaxes(-1, -2)
    return weights


def _host_embedding_table(matrix: FullPrecisionMatrix) -> jax.Array:
    return jnp.asarray(jax.device_get(matrix.weights))


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_export_load_roundtrips_weights_spec_and_template_sharding(
    fake_mesh: Mesh,
    layout: Layout,
) -> None:
    weights = _logical_weights()
    spec = FullPrecisionSpec(layout=layout)
    saved_sharding = make_sharding((None, None))
    original = FullPrecisionMatrix(
        spec=spec,
        sharding_config=make_test_sharding_config(),
        is_sharded=False,
        weights=jax.device_put(_stored_weights(layout, weights), saved_sharding),
    )
    skeleton = spec.compress(
        dummy_array(weights.shape, weights.dtype, make_sharding((None, None))),
        sharding_config=make_test_sharding_config(),
    )

    restored = skeleton.load_exported(original.export())

    assert isinstance(restored, FullPrecisionMatrix)
    assert_named_sharding(skeleton.weights.sharding, fake_mesh)
    assert original.export().metadata["spec"] == spec.to_json()
    assert restored.spec == spec
    assert restored.weights.sharding == skeleton.weights.sharding
    assert restored.weights.sharding != original.weights.sharding
    assert_close_arrays(result=restored.weights, reference=original.weights)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_dot_matches_logical_matmul_and_preserves_input_sharding(
    fake_mesh: Mesh,
    layout: Layout,
) -> None:
    weights = _logical_weights()
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((LogicalAxis.BATCH,)))
    matrix = FullPrecisionSpec(layout=layout).compress(weights, sharding_config=make_test_sharding_config())

    result = matrix.dot(vector, keychain=Keychain.init(0, sharding_config=make_test_sharding_config()))

    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vector.sharding
    assert_close_arrays(result=result, reference=weights @ vector)


def test_initializer_weight_matrix_is_sharded_false_uses_replicated_weight_sharding() -> None:
    initializer = EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config())

    matrix = initializer.weight_matrix(4, 4, is_sharded=False)

    assert matrix.sharding_config == make_test_sharding_config()
    assert matrix.decompress().sharding == make_sharding((None, None))


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_mixture_weight_matrix_shards_only_mixture_axes(layout: Layout) -> None:
    matrix = FullPrecisionSpec(layout=layout).compress(
        _logical_weights(2),
        sharding_config=make_test_sharding_config(),
    )

    assert isinstance(matrix.weights.sharding, NamedSharding)
    assert matrix.weights.sharding.spec == PartitionSpec(LogicalAxis.MIXTURE.value, None, None)


def test_full_precision_lookup_embedding_matches_selected_row_and_feature_sharding(fake_mesh: Mesh) -> None:
    del fake_mesh
    spec = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT)
    matrix = FullPrecisionMatrix(
        spec=spec,
        sharding_config=make_test_sharding_config(),
        is_sharded=False,
        weights=spec.layout.from_output_input(
            _logical_weights(),
            sharding=make_sharding((None, None)),
        ),
    )
    token_index = 2

    result = matrix.lookup_embedding(
        token_index, keychain=Keychain.init(5, sharding_config=make_test_sharding_config())
    )

    assert_close_arrays(result=result, reference=_host_embedding_table(matrix)[token_index, :])
    assert result.sharding == make_sharding((None,))
