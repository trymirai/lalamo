from collections.abc import Callable, Iterable
from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, ShapeDtypeStruct
from jaxtyping import DTypeLike, PyTree

from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import WeightMatrix

__all__ = [
    "load_as",
    "load_as_at",
    "map_nodes_of_type",
    "map_nodes_of_type_with_path",
    "select_nodes_of_type",
    "zip_nodes_with",
]


def _is_weight_matrix(value: PyTree) -> bool:
    return isinstance(value, WeightMatrix)


def _is_array_like(value: PyTree) -> bool:
    return isinstance(value, (Array, ShapeDtypeStruct))


def _path_name(path: tuple[object, ...]) -> str:
    if not path:
        return " at root"
    return f" at path {jtu.keystr(path)}"


def _shape_dtype(value: Array | ShapeDtypeStruct) -> ShapeDtypeStruct:
    # Compatibility checks ignore sharding because values are resharded after dtype/shape validation.
    return ShapeDtypeStruct(value.shape, value.dtype, weak_type=getattr(value, "weak_type", False))


def _check_array_compatible(
    template_leaf: Array | ShapeDtypeStruct,
    value_leaf: Array | ShapeDtypeStruct,
) -> DTypeLike:
    def check_shape_compatibility(template: Array, value: Array) -> Array:
        # Stack requires exact shape equality while still using JAX's dtype promotion rules.
        return jnp.stack([value, template])

    with jax.numpy_dtype_promotion("strict"):
        return jax.eval_shape(check_shape_compatibility, _shape_dtype(template_leaf), _shape_dtype(value_leaf)).dtype


def _check_weight_matrix_compatible(
    path: tuple[object, ...],
    template_leaf: WeightMatrix,
    value_leaf: WeightMatrix,
) -> None:
    if template_leaf.shape != value_leaf.shape:
        raise ValueError(
            f"Expected parameter{_path_name(path)} to have shape {template_leaf.shape}, got {value_leaf.shape}",
        )
    if template_leaf.dtype != value_leaf.dtype:
        raise ValueError(
            f"Expected parameter{_path_name(path)} to have dtype {template_leaf.dtype}, got {value_leaf.dtype}",
        )


def _check_leaf_compatible(
    path: tuple[object, ...],
    template_leaf: PyTree,
    value_leaf: PyTree,
) -> None:
    if isinstance(template_leaf, WeightMatrix) and isinstance(value_leaf, WeightMatrix):
        _check_weight_matrix_compatible(path, template_leaf, value_leaf)
        return
    if _is_array_like(template_leaf) and _is_array_like(value_leaf):
        _check_array_compatible(template_leaf, value_leaf)
        return
    if type(template_leaf) is not type(value_leaf):
        raise TypeError(
            f"Expected parameter{_path_name(path)} to have type {type(template_leaf)}, got {type(value_leaf)}",
        )


def _load_leaf_as_template(template_leaf: PyTree, value_leaf: PyTree) -> PyTree:
    if isinstance(template_leaf, WeightMatrix) and isinstance(value_leaf, WeightMatrix):
        return value_leaf.switch_sharding_config(template_leaf.sharding_config)
    if _is_array_like(template_leaf) and _is_array_like(value_leaf):
        template_sharding = template_leaf.sharding
        assert template_sharding is not None
        dtype = _check_array_compatible(template_leaf, value_leaf)
        if isinstance(value_leaf, ShapeDtypeStruct):
            return dummy_array(value_leaf.shape, dtype, template_sharding)
        return jax.device_put(value_leaf.astype(dtype), template_sharding)
    return value_leaf


def _check_compatible(
    old_value: PyTree,
    new_value: PyTree,
    parent_node: PyTree | None = None,  # noqa: ARG001
) -> None:
    template_leaves_with_paths, template_tree_def = jtu.tree_flatten_with_path(old_value, is_leaf=_is_weight_matrix)
    value_leaves, value_tree_def = jtu.tree_flatten(new_value, is_leaf=_is_weight_matrix)
    if template_tree_def != value_tree_def:
        raise TypeError(
            f"Tried to load a value of shape {value_tree_def} into an incompatible pytree of shape {template_tree_def}"
        )

    for (path, template_leaf), value_leaf in zip(template_leaves_with_paths, value_leaves, strict=True):
        _check_leaf_compatible(path, template_leaf, value_leaf)


@overload
def load_as[ValueT: WeightMatrix](
    template: WeightMatrix,
    value: ValueT,
) -> ValueT: ...


@overload
def load_as[TreeT](
    template: TreeT,
    value: TreeT,
) -> TreeT: ...


def load_as(template: PyTree, value: PyTree) -> PyTree:
    _check_compatible(template, value)

    template_leaves_with_paths, template_tree_def = jtu.tree_flatten_with_path(template, is_leaf=_is_weight_matrix)
    value_leaves, _ = jtu.tree_flatten(value, is_leaf=_is_weight_matrix)

    result_leaves = [
        _load_leaf_as_template(template_leaf, value_leaf)
        for (_path, template_leaf), value_leaf in zip(template_leaves_with_paths, value_leaves, strict=True)
    ]

    return jtu.tree_unflatten(template_tree_def, result_leaves)


def load_as_at[TreeT: PyTree](
    selector: Callable[[TreeT], Iterable[PyTree]],
    tree: TreeT,
    values: Iterable[PyTree],
) -> TreeT:
    old_values = list(selector(tree))
    new_values = list(values)

    loaded_new_values = tuple(
        load_as(old_value, new_value) for old_value, new_value in zip(old_values, new_values, strict=True)
    )

    return eqx.tree_at(selector, tree, loaded_new_values, is_leaf=lambda value: value is None)


def map_nodes_of_type[
    TreeT: PyTree,
    NodeT,
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


def map_nodes_of_type_with_path[TreeT: PyTree, NodeT](
    node_type: type[NodeT],
    map_fn: Callable[[tuple[str, ...], NodeT], PyTree],
    tree: TreeT,
) -> TreeT:
    def wrapper(path: tuple[object, ...], node: PyTree) -> PyTree:
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


def zip_nodes_with[TreeT: PyTree](
    map_fn: Callable[..., PyTree],
    first_tree: TreeT,
    *trees: TreeT,
    is_leaf: Callable[[PyTree], bool] | None = None,
    leaf_dtype: type | tuple[type, ...] | None = None,
) -> TreeT:
    should_map_all_nones = is_leaf is not None and is_leaf(None)

    def is_leaf_wrapper(leaf: PyTree) -> bool:
        if leaf is None:
            return True
        if leaf_dtype is not None and isinstance(leaf, leaf_dtype):
            return True
        if is_leaf is None:
            return False
        return is_leaf(leaf)

    def wrapper(*leaves: PyTree) -> PyTree:
        if not should_map_all_nones and all(leaf is None for leaf in leaves):
            return None
        return map_fn(*leaves)

    return jax.tree.map(wrapper, first_tree, *trees, is_leaf=is_leaf_wrapper)
