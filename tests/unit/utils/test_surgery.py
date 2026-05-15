from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding
from jaxtyping import DTypeLike, PyTree

from lalamo.utils.surgery import (
    load_as,
    map_nodes_of_type,
    zip_nodes_with,
)
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array


class Block(eqx.Module):
    matrix: FullPrecisionMatrix
    linears: tuple[Linear, ...]


def _matrix(
    shape: tuple[int, int] = (2, 3),
    dtype: DTypeLike = jnp.float32,
    fill_value: float = 0,
) -> FullPrecisionMatrix:
    return FullPrecisionSpec().compress(jnp.full(shape, fill_value, dtype=dtype), is_sharded=False)


def test_load_as_loads_matching_array_tree() -> None:
    template = {"linear": Linear(weight=jnp.zeros((2, 3)), bias=jnp.zeros((2,)))}
    value = {"linear": Linear(weight=jnp.ones((2, 3)), bias=jnp.ones((2,)))}

    result = load_as(template, value)

    assert jnp.all(result["linear"].weight == 1)
    assert jnp.all(result["linear"].bias == 1)


@pytest.mark.parametrize(
    ("template", "value", "error", "match"),
    [
        ({"weight": jnp.zeros((2, 3))}, {"weight": jnp.ones((2, 3)), "bias": jnp.ones((2,))}, TypeError, "pytree"),
        (jnp.zeros((2, 3), dtype=jnp.float32), jnp.ones((2, 4), dtype=jnp.float32), ValueError, "shape"),
        (jnp.zeros((2, 3), dtype=jnp.float32), jnp.ones((2, 3), dtype=jnp.float16), ValueError, "dtype"),
        ({"name": "linear"}, {"name": 1}, TypeError, "type"),
        (_matrix(shape=(2, 3)), _matrix(shape=(2, 4), fill_value=1), ValueError, "shape"),
        (_matrix(dtype=jnp.float32), _matrix(dtype=jnp.float16, fill_value=1), ValueError, "dtype"),
    ],
)
def test_load_as_rejects_mismatches(template: PyTree, value: PyTree, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        load_as(template, value)


def test_load_as_casts_array_dtype_when_allowed() -> None:
    result = load_as(jnp.zeros((2, 3), dtype=jnp.float32), jnp.ones((2, 3), dtype=jnp.float16), allow_dtype_cast=True)

    assert result.dtype == jnp.float32
    assert jnp.all(result == 1)


def test_load_as_treats_weight_matrices_as_leaf_nodes() -> None:
    template = {"matrix": _matrix()}
    value = {"matrix": _matrix(fill_value=1)}

    result = load_as(template, value)

    assert result["matrix"] is value["matrix"]


def test_load_as_switches_weight_matrix_sharding_to_template(fake_mesh: Mesh) -> None:
    template = FullPrecisionSpec().compress(jnp.zeros((2, 3), dtype=jnp.float32))
    value = FullPrecisionSpec().compress(jnp.ones((2, 3), dtype=jnp.float32), is_sharded=False)

    result = load_as(template, value)

    assert result.is_sharded == template.is_sharded
    assert result.weights.sharding == template.weights.sharding
    assert isinstance(result.weights.sharding, NamedSharding)
    assert result.weights.sharding.mesh == fake_mesh
    assert jnp.all(result.weights == 1)


def test_load_as_casts_weight_matrix_dtype_when_allowed() -> None:
    template = _matrix(dtype=jnp.float32)
    value = _matrix(dtype=jnp.float16, fill_value=1)

    result = load_as(template, value, allow_dtype_cast=True)

    assert isinstance(result, FullPrecisionMatrix)
    assert result.dtype == jnp.float32
    assert jnp.all(result.weights == 1)


def test_map_nodes_of_type_updates_only_selected_nodes() -> None:
    block = Block(
        matrix=_matrix(fill_value=1),
        linears=(
            Linear(weight=jnp.ones((2, 2)), bias=jnp.ones((2,))),
            Linear(weight=jnp.full((2, 2), 2), bias=jnp.full((2,), 2)),
        ),
    )

    result = map_nodes_of_type(
        Linear,
        lambda linear: Linear(weight=linear.weight + 1, bias=linear.bias),
        block,
    )

    assert isinstance(result.matrix, FullPrecisionMatrix)
    assert jnp.array_equal(result.matrix.weights, block.matrix.weights)
    assert jnp.all(result.linears[0].weight == 2)
    assert jnp.all(result.linears[1].weight == 3)
    assert jnp.all(result.linears[0].bias == 1)


def test_zip_nodes_with_updates_matching_leaves() -> None:
    first_block = Block(
        matrix=_matrix(fill_value=1),
        linears=(
            Linear(weight=jnp.ones((2, 2)), bias=jnp.ones((2,))),
            Linear(weight=jnp.full((2, 2), 2), bias=jnp.full((2,), 2)),
        ),
    )
    second_block = Block(
        matrix=_matrix(fill_value=1),
        linears=(
            Linear(weight=jnp.full((2, 2), 3), bias=jnp.full((2,), 3)),
            Linear(weight=jnp.full((2, 2), 4), bias=jnp.full((2,), 4)),
        ),
    )

    def add_linears(first_leaf: PyTree, second_leaf: PyTree) -> PyTree:
        if isinstance(first_leaf, Linear):
            return Linear(
                weight=first_leaf.weight + cast("Linear", second_leaf).weight,
                bias=first_leaf.bias,
            )
        return first_leaf

    result = zip_nodes_with(
        add_linears,
        first_block,
        second_block,
        is_leaf=lambda leaf: isinstance(leaf, FullPrecisionMatrix),
        leaf_dtype=Linear,
    )

    assert isinstance(result, Block)
    assert jnp.array_equal(result.matrix.weights, first_block.matrix.weights)
    assert jnp.array_equal(result.linears[0].weight, jnp.full((2, 2), 4))
    assert jnp.array_equal(result.linears[1].weight, jnp.full((2, 2), 6))
    assert jnp.array_equal(result.linears[0].bias, first_block.linears[0].bias)


def test_zip_nodes_with_rejects_mismatched_paths() -> None:
    first_block = Block(
        matrix=_matrix(fill_value=1),
        linears=(
            Linear(weight=jnp.ones((2, 2)), bias=jnp.ones((2,))),
            Linear(weight=jnp.full((2, 2), 2), bias=jnp.full((2,), 2)),
        ),
    )
    second_block = Block(
        matrix=_matrix(fill_value=1),
        linears=(Linear(weight=jnp.full((2, 2), 3), bias=jnp.full((2,), 3)),),
    )

    def keep_leaf(first_leaf: PyTree, _second_leaf: PyTree) -> PyTree:
        return first_leaf

    with pytest.raises(ValueError):
        zip_nodes_with(
            keep_leaf,
            first_block,
            second_block,
            is_leaf=lambda leaf: isinstance(leaf, Linear),
            leaf_dtype=Linear,
        )
