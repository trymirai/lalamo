from collections.abc import Callable
from functools import wraps
from typing import TypeGuard, cast

import equinox as eqx
import jax.tree_util as jtu
from jax import Array as JaxArray
from jax import ShapeDtypeStruct
from jax.sharding import NamedSharding, Sharding
from jaxtyping import Array, DTypeLike

from lalamo.utils.sharding import sharding_of, with_sharding

__all__ = [
    "OutShardingRule",
    "contains_dummy_arrays",
    "dummy_array",
    "is_dummy_array",
    "preserve_first_input_sharding",
    "supports_dummy_arrays",
]

type OutShardingRule = Callable[[tuple[NamedSharding, ...]], NamedSharding | None]


def dummy_array(shape: int | tuple[int, ...], dtype: DTypeLike, sharding: Sharding | None = None) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    return cast("Array", ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding))


def preserve_first_input_sharding(input_shardings: tuple[NamedSharding, ...]) -> NamedSharding | None:
    if not input_shardings:
        return None
    return input_shardings[0]


def is_dummy_array(value: object) -> TypeGuard[Array]:
    return isinstance(value, ShapeDtypeStruct)


def contains_dummy_arrays(value: object) -> bool:
    return any(is_dummy_array(leaf) for leaf in jtu.tree_leaves(value))


def _input_named_shardings(value: object) -> tuple[NamedSharding, ...]:
    shardings = []
    for leaf in jtu.tree_leaves(value):
        if not is_dummy_array(leaf) and not isinstance(leaf, JaxArray):
            continue

        sharding = sharding_of(leaf)
        if isinstance(sharding, NamedSharding):
            shardings.append(sharding)
    return tuple(shardings)


def _apply_dummy_array_sharding(value: object, sharding: NamedSharding | None) -> object:
    if is_dummy_array(value):
        return with_sharding(value, sharding)
    return value


def _concretize_dummy_array_mesh(value: object, input_shardings: tuple[NamedSharding, ...]) -> object:
    if not input_shardings or not is_dummy_array(value):
        return value

    sharding = sharding_of(value)
    if not isinstance(sharding, NamedSharding):
        return value
    return with_sharding(value, NamedSharding(input_shardings[0].mesh, sharding.spec))


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
            return cast("ResultT", jtu.tree_map(lambda leaf: _apply_dummy_array_sharding(leaf, sharding), result))

        return wrapped

    return decorator
