from collections.abc import Callable
from functools import wraps
from typing import TypeGuard, cast

import equinox as eqx
import jax.tree_util as jtu
from jax import Array as JaxArray
from jax import ShapeDtypeStruct
from jax.sharding import Mesh, NamedSharding
from jaxtyping import Array, DTypeLike

__all__ = [
    "OutShardingRule",
    "contains_dummy_arrays",
    "dummy_array",
    "is_dummy_array",
    "preserve_first_input_sharding",
    "supports_dummy_arrays",
]

type OutShardingRule = Callable[[tuple[NamedSharding, ...]], NamedSharding]


def dummy_array(
    shape: int | tuple[int, ...],
    dtype: DTypeLike,
    sharding: NamedSharding,
) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    return cast("Array", ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding))


def preserve_first_input_sharding(input_shardings: tuple[NamedSharding, ...]) -> NamedSharding:
    if not input_shardings:
        raise ValueError("Cannot preserve the first input sharding without any sharded inputs.")
    first_input_sharding, *_ = input_shardings
    return first_input_sharding


def is_dummy_array(value: object) -> TypeGuard[Array]:
    return isinstance(value, ShapeDtypeStruct)


def contains_dummy_arrays(value: object) -> bool:
    return any(is_dummy_array(leaf) for leaf in jtu.tree_leaves(value))


def _input_named_shardings(value: object) -> tuple[NamedSharding, ...]:
    shardings = []
    for leaf in jtu.tree_leaves(value):
        if is_dummy_array(leaf):
            sharding = leaf.sharding
            if not isinstance(sharding, NamedSharding):
                raise TypeError(f"Expected dummy array input with NamedSharding, got {type(sharding).__name__}.")
            if not isinstance(sharding.mesh, Mesh):
                raise TypeError(f"Expected dummy array input with concrete Mesh, got {type(sharding.mesh).__name__}.")
            if sharding.mesh.empty:
                raise ValueError("Expected dummy array input with non-empty mesh.")
            shardings.append(sharding)
            continue

        if isinstance(leaf, JaxArray):
            sharding = leaf.sharding
            if isinstance(sharding, NamedSharding) and isinstance(sharding.mesh, Mesh) and not sharding.mesh.empty:
                shardings.append(sharding)
    return tuple(shardings)


def _apply_dummy_array_sharding(
    value: object,
    sharding: NamedSharding,
) -> object:
    if not is_dummy_array(value):
        return value
    return dummy_array(value.shape, value.dtype, sharding)


def _concretize_dummy_array_mesh(value: object, input_shardings: tuple[NamedSharding, ...]) -> object:
    if not is_dummy_array(value):
        return value

    sharding = value.sharding
    if isinstance(sharding, NamedSharding) and isinstance(sharding.mesh, Mesh) and not sharding.mesh.empty:
        return value
    if not isinstance(sharding, NamedSharding):
        raise TypeError(f"Expected dummy array output with NamedSharding, got {type(sharding).__name__}.")
    first_input_sharding, *_ = input_shardings
    return dummy_array(value.shape, value.dtype, NamedSharding(first_input_sharding.mesh, sharding.spec))


def supports_dummy_arrays[**Params, ResultT](
    *,
    out_sharding_rule: OutShardingRule | None = None,
) -> Callable[[Callable[Params, ResultT]], Callable[Params, ResultT]]:
    def decorator(function: Callable[Params, ResultT]) -> Callable[Params, ResultT]:
        @wraps(function)
        def wrapped(*args: Params.args, **kwargs: Params.kwargs) -> ResultT:
            inputs = (args, kwargs)
            if not contains_dummy_arrays(inputs):
                return function(*args, **kwargs)

            result = eqx.filter_eval_shape(function, *args, **kwargs)
            input_shardings = _input_named_shardings(inputs)
            if out_sharding_rule is None:
                return cast(
                    "ResultT",
                    jtu.tree_map(lambda leaf: _concretize_dummy_array_mesh(leaf, input_shardings), result),
                )

            sharding = out_sharding_rule(input_shardings)
            return cast(
                "ResultT",
                jtu.tree_map(lambda leaf: _apply_dummy_array_sharding(leaf, sharding), result),
            )

        return wrapped

    return decorator
