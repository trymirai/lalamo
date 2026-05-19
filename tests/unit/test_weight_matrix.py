from math import prod

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding

from lalamo.module import Keychain, ShardingAxis
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import (
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
)
from tests.common import assert_close_arrays, assert_named_sharding


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 4)
    return jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) / 10


def _stored_weights(layout: Layout, weights: jax.Array) -> jax.Array:
    if layout == Layout.INPUT_OUTPUT:
        return weights.swapaxes(-1, -2)
    return weights


def _host_embedding_table(matrix: FullPrecisionMatrix) -> jax.Array:
    return jnp.asarray(jax.device_get(matrix.weights))


def _embedding_sharding() -> NamedSharding:
    sharding = make_sharding((None,))
    assert sharding is not None
    return sharding


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
        is_sharded=False,
        weights=jax.device_put(_stored_weights(layout, weights), saved_sharding),
    )
    skeleton = spec.compress(dummy_array(weights.shape, weights.dtype))

    restored = skeleton.load_exported(original.export())

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
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,)))
    matrix = FullPrecisionSpec(layout=layout).compress(weights)

    result = matrix.dot(vector, keychain=Keychain.init(0))

    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vector.sharding
    assert_close_arrays(result=result, reference=weights @ vector)


def test_full_precision_lookup_embedding_matches_selected_row_and_feature_sharding(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_index = 2

    result = matrix.lookup_embedding(token_index, keychain=Keychain.init(5))

    assert_close_arrays(result=result, reference=_host_embedding_table(matrix)[token_index, :])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _embedding_sharding()
