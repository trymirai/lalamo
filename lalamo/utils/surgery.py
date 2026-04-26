from collections.abc import Callable, Iterable
from typing import overload

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax import Array, ShapeDtypeStruct
from jaxtyping import DTypeLike, PyTree

from lalamo.weight_matrix import WeightMatrix

__all__ = [
    "load_as",
    "load_as_at",
    "map_nodes_of_type",
    "map_nodes_of_type_with_path",
    "select_nodes_of_type",
]

type ArrayLike = Array | ShapeDtypeStruct


def _is_weight_matrix(value: PyTree) -> bool:
    return isinstance(value, WeightMatrix)


def _is_array_like(value: PyTree) -> bool:
    return isinstance(value, (Array, ShapeDtypeStruct))


def _path_name(path: tuple[object, ...]) -> str:
    if not path:
        return " at root"
    return f" at path {jtu.keystr(path)}"


def _astype_array_like(value: ArrayLike, dtype: DTypeLike) -> ArrayLike:
    if isinstance(value, ShapeDtypeStruct):
        return ShapeDtypeStruct(shape=value.shape, dtype=dtype, sharding=value.sharding)
    return value.astype(dtype)


def _check_array_compatible(
    path: tuple[object, ...],
    template_leaf: ArrayLike,
    value_leaf: ArrayLike,
    *,
    allow_dtype_cast: bool,
) -> None:
    if template_leaf.shape != value_leaf.shape:
        raise ValueError(
            f"Expected parameter{_path_name(path)} to have shape {template_leaf.shape}, got {value_leaf.shape}",
        )
    if not allow_dtype_cast and template_leaf.dtype != value_leaf.dtype:
        raise ValueError(
            f"Expected parameter{_path_name(path)} to have dtype {template_leaf.dtype}, got {value_leaf.dtype}",
        )
    if template_leaf.sharding != value_leaf.sharding:
        raise ValueError(
            f"Expected parameter{_path_name(path)} to have sharding {template_leaf.sharding}, "
            f"got {value_leaf.sharding}",
        )


def _check_weight_matrix_compatible(
    path: tuple[object, ...],
    template_leaf: WeightMatrix,
    value_leaf: WeightMatrix,
    *,
    allow_dtype_cast: bool,
) -> None:
    if template_leaf.shape != value_leaf.shape:
        raise ValueError(
            f"Expected WeightMatrix{_path_name(path)} to have shape {template_leaf.shape}, got {value_leaf.shape}",
        )
    if not allow_dtype_cast and template_leaf.dtype != value_leaf.dtype:
        raise ValueError(
            f"Expected WeightMatrix{_path_name(path)} to have dtype {template_leaf.dtype}, got {value_leaf.dtype}",
        )


def _check_leaf_compatible(
    path: tuple[object, ...],
    template_leaf: PyTree,
    value_leaf: PyTree,
    *,
    allow_dtype_cast: bool,
) -> None:
    if isinstance(template_leaf, WeightMatrix) and isinstance(value_leaf, WeightMatrix):
        _check_weight_matrix_compatible(path, template_leaf, value_leaf, allow_dtype_cast=allow_dtype_cast)
        return
    if _is_array_like(template_leaf) and _is_array_like(value_leaf):
        _check_array_compatible(path, template_leaf, value_leaf, allow_dtype_cast=allow_dtype_cast)
        return
    if type(template_leaf) is not type(value_leaf):
        raise TypeError(
            f"Expected parameter{_path_name(path)} to have type {type(template_leaf)}, got {type(value_leaf)}",
        )


def _cast_leaf_dtype(template_leaf: PyTree, value_leaf: PyTree) -> PyTree:
    if isinstance(template_leaf, WeightMatrix) and isinstance(value_leaf, WeightMatrix):
        if template_leaf.dtype == value_leaf.dtype:
            return value_leaf
        return value_leaf.astype(template_leaf.dtype)
    if _is_array_like(template_leaf) and _is_array_like(value_leaf):
        if template_leaf.dtype == value_leaf.dtype:
            return value_leaf
        return _astype_array_like(value_leaf, template_leaf.dtype)
    return value_leaf


def _check_compatible(
    old_value: PyTree,
    new_value: PyTree,
    parent_node: PyTree | None = None,  # noqa: ARG001
    *,
    allow_dtype_cast: bool = False,
) -> None:
    template_leaves_with_paths, template_tree_def = jtu.tree_flatten_with_path(old_value, is_leaf=_is_weight_matrix)
    value_leaves, value_tree_def = jtu.tree_flatten(new_value, is_leaf=_is_weight_matrix)
    if template_tree_def != value_tree_def:
        raise TypeError(
            f"Tried to load a value of shape {value_tree_def} into an incompatible pytree of shape {template_tree_def}"
        )

    for (path, template_leaf), value_leaf in zip(template_leaves_with_paths, value_leaves, strict=True):
        _check_leaf_compatible(path, template_leaf, value_leaf, allow_dtype_cast=allow_dtype_cast)


@overload
def load_as[ValueT: WeightMatrix](
    template: WeightMatrix,
    value: ValueT,
    allow_dtype_cast: bool = False,
) -> ValueT: ...


@overload
def load_as[TreeT](template: TreeT, value: TreeT, allow_dtype_cast: bool = False) -> TreeT: ...


def load_as(template: PyTree, value: PyTree, allow_dtype_cast: bool = False) -> PyTree:
    _check_compatible(template, value, allow_dtype_cast=allow_dtype_cast)

    template_leaves_with_paths, template_tree_def = jtu.tree_flatten_with_path(template, is_leaf=_is_weight_matrix)
    value_leaves, _ = jtu.tree_flatten(value, is_leaf=_is_weight_matrix)

    result_leaves = [
        _cast_leaf_dtype(template_leaf, value_leaf)
        for (_path, template_leaf), value_leaf in zip(template_leaves_with_paths, value_leaves, strict=True)
    ]

    return jtu.tree_unflatten(template_tree_def, result_leaves)


def load_as_at[TreeT: PyTree](
    selector: Callable[[TreeT], Iterable[PyTree]],
    tree: TreeT,
    values: Iterable[PyTree],
    allow_dtype_cast: bool = False,
) -> TreeT:
    old_values = list(selector(tree))
    new_values = list(values)

    loaded_new_values = tuple(
        load_as(old_value, new_value, allow_dtype_cast)
        for old_value, new_value in zip(old_values, new_values, strict=True)
    )

    return eqx.tree_at(selector, tree, loaded_new_values)


def map_nodes_of_type[
    TreeT: PyTree,
    NodeT: PyTree,
](
    node_type: type[NodeT],
    map_fn: Callable[[NodeT], NodeT],
    tree: TreeT,
) -> TreeT:
    def wrapper(node: PyTree) -> PyTree:
        if isinstance(node, node_type):
            return map_fn(node)
        return node

    return jax.tree.map(wrapper, tree, is_leaf=lambda leaf: isinstance(leaf, node_type))


def map_nodes_of_type_with_path[TreeT: PyTree, NodeT: PyTree](
    node_type: type[NodeT],
    map_fn: Callable[[tuple[str, ...], NodeT], NodeT],
    tree: TreeT,
) -> TreeT:
    def wrapper(path: tuple[object, ...], node: eqx.Module) -> eqx.Module:
        if isinstance(node, node_type):
            return map_fn(tuple(map(str, path)), node)
        return node

    return jax.tree.map_with_path(wrapper, tree, is_leaf=lambda leaf: isinstance(leaf, node_type))


def select_nodes_of_type[NodeT: eqx.Module](
    node_type: type[NodeT],
    tree: PyTree,
) -> list[tuple[tuple[str, ...], NodeT]]:
    return [
        (tuple(map(str, path)), leaf)
        for (path, leaf) in jax.tree.leaves_with_path(tree, is_leaf=lambda node: isinstance(node, node_type))
        if isinstance(leaf, node_type)
    ]
