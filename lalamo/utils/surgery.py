from collections.abc import Callable, Iterable

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax import ShapeDtypeStruct
from jaxtyping import Array, PyTree

__all__ = [
    "load_as",
    "load_as_at",
    "map_nodes_of_type",
    "map_nodes_of_type_with_path",
    "select_nodes_of_type",
]


def _get_name(leaf: PyTree, tree: PyTree) -> str:
    for path, value in jtu.tree_leaves_with_path(tree):
        if value is leaf:
            return f"~{jtu.keystr(path)}"
    raise ValueError(f"Leaf {leaf} not found in tree {tree}")


def _check_compatible(old_value: PyTree, new_value: PyTree, parent_node: PyTree | None = None) -> None:
    if isinstance(old_value, (Array, ShapeDtypeStruct)) and isinstance(new_value, Array):
        if parent_node is not None:
            full_name = f" {parent_node}.{_get_name(old_value, parent_node)}"
        else:
            full_name = ""
        if old_value.shape != new_value.shape:
            raise ValueError(
                f"Expected parameter{full_name} to have shape {old_value.shape}, got {new_value.shape}",
            )
        if old_value.dtype != new_value.dtype:
            raise ValueError(
                f"Expected parameter{full_name} to have dtype {old_value.dtype}, got {new_value.dtype}",
            )
    elif type(old_value) is not type(new_value):
        raise TypeError(f"Expected parameter of type {type(old_value)}, got {type(new_value)}")


def load_as[TreeT: PyTree](template: TreeT, value: TreeT, allow_dtype_cast: bool = False) -> TreeT:
    template_leaves_with_paths, template_tree_def = jtu.tree_flatten_with_path(template)
    value_leaves, value_tree_def = jtu.tree_flatten(value)
    if template_tree_def != value_tree_def:
        raise TypeError(
            f"Tried to load a value of shape {value_tree_def} into an incompatible pytree of shape {template_tree_def}"
        )

    result_leaves = []
    for (path, template_leaf), value_leaf in zip(template_leaves_with_paths, value_leaves, strict=True):
        result_leaf = value_leaf
        if result_leaf.dtype != template_leaf.dtype:
            if not allow_dtype_cast:
                raise ValueError(f"Expected dtype {template_leaf.dtype} at path {path}, got {result_leaf.dtype}")
            result_leaf = result_leaf.astype(template_leaf.dtype)
        if result_leaf.shape != template_leaf.shape:
            raise ValueError(f"Expected shape {template_leaf.shape} at path {path}, got {result_leaf.shape}")
        if result_leaf.sharding != template_leaf.sharding:
            result_leaf = jax.device_put(value_leaf, template_leaf.sharding)
        result_leaves.append(result_leaf)

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
