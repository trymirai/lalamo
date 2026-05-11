from dataclasses import dataclass
from typing import Literal, Self, cast, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float, Key, PyTree

from lalamo.initializer import Initializer
from lalamo.module import Keychain, ShardingAxis
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.utils.surgery import (
    load_as,
    map_nodes_of_type,
    map_nodes_of_type_with_path,
    select_nodes_of_type,
    zip_nodes_with,
)
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    ShapeDtypeMatrix,
    ShapeDtypeSpec,
    WeightMatrix,
    WeightMatrixSpec,
)


@dataclass(frozen=True)
class SurgeryWeightMatrixSpec(WeightMatrixSpec):
    def compress(
        self,
        weights: Float[Array, "*batch out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> WeightMatrix:
        del weights, key, preconditioner, implementation, is_sharded
        raise NotImplementedError

    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> WeightMatrix:
        raise NotImplementedError


class SurgeryWeightMatrix(WeightMatrix[SurgeryWeightMatrixSpec]):
    weights: Float[Array, "out_channels in_channels"]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return type(self)(spec=self.spec, is_sharded=self.is_sharded, weights=self.weights.astype(dtype))

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionMatrix(spec=FullPrecisionSpec(), is_sharded=self.is_sharded, weights=self.weights)

    def decompress(self) -> Float[Array, "out_channels in_channels"]:
        return self.weights

    @overload
    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: Literal[False] = False,
    ) -> Float[Array, " out_channels"]: ...

    @overload
    def dot(
        self,
        vector: Float[Array, " out_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: Literal[True],
    ) -> Float[Array, " in_channels"]: ...

    def dot(
        self,
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
        transposed: bool = False,
    ) -> Float[Array, " channels"]:
        weights = self.weights
        if transposed:
            weights = weights.T
        return weights @ vector


class TemplateWeightMatrix(SurgeryWeightMatrix):
    pass


class ValueWeightMatrix(SurgeryWeightMatrix):
    pass


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array


class Block(eqx.Module):
    matrix: SurgeryWeightMatrix
    linears: tuple[Linear, ...]


def _template_matrix(
    shape: tuple[int, int] = (2, 3),
    dtype: DTypeLike = jnp.float32,
) -> TemplateWeightMatrix:
    return TemplateWeightMatrix(
        spec=SurgeryWeightMatrixSpec(),
        is_sharded=False,
        weights=jnp.zeros(shape, dtype=dtype),
    )


def _value_matrix(
    shape: tuple[int, int] = (2, 3),
    dtype: DTypeLike = jnp.float32,
) -> ValueWeightMatrix:
    return ValueWeightMatrix(spec=SurgeryWeightMatrixSpec(), is_sharded=False, weights=jnp.ones(shape, dtype=dtype))


def test_load_as_loads_matching_array_tree() -> None:
    template = {"linear": Linear(weight=jnp.zeros((2, 3)), bias=jnp.zeros((2,)))}
    value = {"linear": Linear(weight=jnp.ones((2, 3)), bias=jnp.ones((2,)))}

    result = load_as(template, value)

    assert jnp.all(result["linear"].weight == 1)
    assert jnp.all(result["linear"].bias == 1)


def test_load_as_rejects_tree_structure_mismatch() -> None:
    template = {"weight": jnp.zeros((2, 3))}
    value = {"weight": jnp.ones((2, 3)), "bias": jnp.ones((2,))}

    with pytest.raises(TypeError, match="incompatible pytree"):
        load_as(template, value)


def test_load_as_rejects_array_shape_mismatch() -> None:
    template = jnp.zeros((2, 3), dtype=jnp.float32)
    value = jnp.ones((2, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="shape"):
        load_as(template, value)


def test_load_as_rejects_array_dtype_mismatch() -> None:
    template = jnp.zeros((2, 3), dtype=jnp.float32)
    value = jnp.ones((2, 3), dtype=jnp.float16)

    with pytest.raises(ValueError, match="dtype"):
        load_as(template, value)


def test_load_as_casts_array_dtype_when_allowed() -> None:
    template = jnp.zeros((2, 3), dtype=jnp.float32)
    value = jnp.ones((2, 3), dtype=jnp.float16)

    result = load_as(template, value, allow_dtype_cast=True)

    assert result.dtype == jnp.float32
    assert jnp.all(result == 1)


def test_load_as_rejects_non_array_leaf_type_mismatch() -> None:
    template = {"name": "linear"}
    value = {"name": 1}

    with pytest.raises(TypeError, match="type"):
        load_as(template, value)


def test_load_as_accepts_matching_shape_dtype_struct() -> None:
    template = dummy_array((2, 3), jnp.float32)
    value = dummy_array((2, 3), jnp.float32)

    result = load_as(template, value)

    assert isinstance(result, ShapeDtypeStruct)
    assert result.shape == value.shape
    assert result.dtype == value.dtype
    assert result.sharding == template.sharding


def test_load_as_casts_shape_dtype_struct_when_allowed() -> None:
    template = dummy_array((2, 3), jnp.float32)
    value = dummy_array((2, 3), jnp.float16)

    result = load_as(template, value, allow_dtype_cast=True)

    assert isinstance(result, ShapeDtypeStruct)
    assert result.shape == value.shape
    assert result.dtype == jnp.float32
    assert result.sharding == value.sharding


def test_load_as_places_array_on_shape_dtype_struct_template_sharding(fake_mesh: object) -> None:
    assert fake_mesh is not None
    template_sharding = make_sharding((ShardingAxis.DATA, None))
    template = dummy_array((2, 3), jnp.float32, template_sharding)
    value = jnp.ones((2, 3), dtype=jnp.float32)

    result = load_as(template, value)

    assert isinstance(result, jax.Array)
    assert result.sharding == template_sharding
    assert jnp.array_equal(result, value)


def test_load_as_treats_weight_matrices_as_leaf_nodes() -> None:
    template = {"matrix": _template_matrix()}
    value = {"matrix": _value_matrix()}

    result = load_as(template, value)

    assert result["matrix"] is value["matrix"]


def test_load_as_accepts_shape_dtype_template_for_input_output_storage_shape() -> None:
    template = ShapeDtypeSpec(layout=Layout.INPUT_OUTPUT).compress(dummy_array((4, 5), jnp.float32))
    value = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(jnp.ones((4, 5), dtype=jnp.float32))

    result = load_as(template, value)

    assert isinstance(template, ShapeDtypeMatrix)
    assert result is value


def test_load_as_rejects_weight_matrix_shape_mismatch() -> None:
    template = _template_matrix(shape=(2, 3))
    value = _value_matrix(shape=(2, 4))

    with pytest.raises(ValueError, match="shape"):
        load_as(template, value)


def test_load_as_rejects_weight_matrix_dtype_mismatch() -> None:
    template = _template_matrix(dtype=jnp.float32)
    value = _value_matrix(dtype=jnp.float16)

    with pytest.raises(ValueError, match="dtype"):
        load_as(template, value)


def test_load_as_casts_weight_matrix_dtype_when_allowed() -> None:
    template = _template_matrix(dtype=jnp.float32)
    value = _value_matrix(dtype=jnp.float16)

    result = load_as(template, value, allow_dtype_cast=True)

    assert isinstance(result, ValueWeightMatrix)
    assert result.dtype == jnp.float32
    assert jnp.all(result.weights == 1)


def test_map_nodes_of_type_updates_only_selected_nodes() -> None:
    block = Block(
        matrix=_value_matrix(),
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

    assert isinstance(result.matrix, ValueWeightMatrix)
    assert jnp.array_equal(result.matrix.weights, block.matrix.weights)
    assert jnp.all(result.linears[0].weight == 2)
    assert jnp.all(result.linears[1].weight == 3)
    assert jnp.all(result.linears[0].bias == 1)


def test_map_nodes_of_type_with_path_passes_node_path() -> None:
    block = Block(
        matrix=_value_matrix(),
        linears=(Linear(weight=jnp.ones((2, 2)), bias=jnp.ones((2,))),),
    )

    result = map_nodes_of_type_with_path(
        Linear,
        lambda path, linear: Linear(weight=linear.weight, bias=jnp.array([len(path)])),
        block,
    )

    assert jnp.array_equal(result.linears[0].bias, jnp.array([2]))


def test_select_nodes_of_type_returns_selected_nodes_with_paths() -> None:
    block = Block(
        matrix=_value_matrix(),
        linears=(
            Linear(weight=jnp.ones((2, 2)), bias=jnp.ones((2,))),
            Linear(weight=jnp.full((2, 2), 2), bias=jnp.full((2,), 2)),
        ),
    )

    selected_nodes = select_nodes_of_type(Linear, block)

    assert [(path, node.weight[0, 0].item()) for path, node in selected_nodes] == [
        ((".linears", "[0]"), 1.0),
        ((".linears", "[1]"), 2.0),
    ]


def test_zip_nodes_with_updates_matching_leaves() -> None:
    first_block = Block(
        matrix=_value_matrix(),
        linears=(
            Linear(weight=jnp.ones((2, 2)), bias=jnp.ones((2,))),
            Linear(weight=jnp.full((2, 2), 2), bias=jnp.full((2,), 2)),
        ),
    )
    second_block = Block(
        matrix=_value_matrix(),
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
        is_leaf=lambda leaf: isinstance(leaf, SurgeryWeightMatrix),
        leaf_dtype=Linear,
    )

    assert isinstance(result, Block)
    assert jnp.array_equal(result.matrix.weights, first_block.matrix.weights)
    assert jnp.array_equal(result.linears[0].weight, jnp.full((2, 2), 4))
    assert jnp.array_equal(result.linears[1].weight, jnp.full((2, 2), 6))
    assert jnp.array_equal(result.linears[0].bias, first_block.linears[0].bias)


def test_zip_nodes_with_skips_all_none_leaves() -> None:
    calls: list[tuple[PyTree, PyTree]] = []

    def add_leaves(first_leaf: PyTree, second_leaf: PyTree) -> PyTree:
        calls.append((first_leaf, second_leaf))
        return cast("int", first_leaf) + cast("int", second_leaf)

    result = zip_nodes_with(
        add_leaves,
        {"empty": None, "value": 1},
        {"empty": None, "value": 2},
    )

    assert result == {"empty": None, "value": 3}
    assert calls == [(1, 2)]


def test_zip_nodes_with_maps_all_none_leaves_when_requested() -> None:
    calls: list[tuple[PyTree, PyTree]] = []

    def map_none_leaves(first_leaf: PyTree, second_leaf: PyTree) -> PyTree:
        calls.append((first_leaf, second_leaf))
        return "mapped"

    result = zip_nodes_with(
        map_none_leaves,
        {"empty": None},
        {"empty": None},
        is_leaf=lambda leaf: leaf is None,
    )

    assert result == {"empty": "mapped"}
    assert calls == [(None, None)]


def test_zip_nodes_with_maps_partial_none_leaves() -> None:
    calls: list[tuple[PyTree, PyTree]] = []

    def map_leaves(first_leaf: PyTree, second_leaf: PyTree) -> PyTree:
        calls.append((first_leaf, second_leaf))
        if first_leaf is None:
            return second_leaf
        return first_leaf

    result = zip_nodes_with(
        map_leaves,
        {"value": None},
        {"value": 1},
    )

    assert result == {"value": 1}
    assert calls == [(None, 1)]


def test_zip_nodes_with_rejects_mismatched_paths() -> None:
    first_block = Block(
        matrix=_value_matrix(),
        linears=(
            Linear(weight=jnp.ones((2, 2)), bias=jnp.ones((2,))),
            Linear(weight=jnp.full((2, 2), 2), bias=jnp.full((2,), 2)),
        ),
    )
    second_block = Block(
        matrix=_value_matrix(),
        linears=(
            Linear(weight=jnp.full((2, 2), 3), bias=jnp.full((2,), 3)),
        ),
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
