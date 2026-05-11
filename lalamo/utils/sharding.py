from collections.abc import Callable
from typing import TypeGuard

from jax import Array, ShapeDtypeStruct, reshard, typeof
from jax.sharding import NamedSharding, PartitionSpec, Sharding, auto_axes, get_abstract_mesh, get_mesh

from lalamo.module import ShardingAxis

__all__ = [
    "is_sharded",
    "make_sharding",
    "reshard_as",
    "sharding_of",
    "use_out_sharding",
    "with_sharding",
]


def make_sharding(partition: tuple[ShardingAxis | None, ...] | None) -> NamedSharding | None:
    if partition is None:
        return None
    try:
        mesh = get_mesh()
    except ValueError:
        mesh = get_abstract_mesh()
    if mesh.empty:
        return None
    return NamedSharding(mesh, PartitionSpec(*partition))


def is_sharded(sharding: Sharding | None) -> TypeGuard[NamedSharding]:
    return isinstance(sharding, NamedSharding) and not sharding.mesh.empty


def sharding_of(array: Array | ShapeDtypeStruct) -> Sharding | None:
    try:
        return array.sharding
    except AttributeError:
        return typeof(array).sharding


def with_sharding(array: Array, sharding: Sharding | None) -> Array:
    if isinstance(array, ShapeDtypeStruct):
        from lalamo.utils.dummy_array import dummy_array  # noqa: PLC0415

        return dummy_array(array.shape, array.dtype, sharding)
    if is_sharded(sharding):
        return reshard(array, sharding)
    return array


def reshard_as(array: Array, reference: Array) -> Array:
    return with_sharding(array, sharding_of(reference))


def use_out_sharding[**Params, ResultT](
    sharding: tuple[ShardingAxis | None, ...],
) -> Callable[[Callable[Params, ResultT]], Callable[Params, ResultT]]:
    return auto_axes(out_sharding=PartitionSpec(*sharding))
