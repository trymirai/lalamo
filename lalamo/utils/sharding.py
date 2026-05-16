from typing import TypeGuard

from jax import Array, ShapeDtypeStruct, reshard, typeof
from jax.sharding import NamedSharding, PartitionSpec, Sharding, get_abstract_mesh, get_mesh
from jaxtyping import Int, Shaped

from lalamo.module import ShardingAxis

__all__ = [
    "is_sharded",
    "lookup_sharded_indices",
    "make_sharding",
    "reshard_as",
    "sharding_of",
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
    axes = [axis for axis in partition if axis is not None]
    if axes and all(mesh.shape[axis] == 1 for axis in axes):
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


def lookup_sharded_indices(
    array: Shaped[Array, "rows cols"],
    indices: int | Int[Array, "*batch"],
) -> Shaped[Array, "*batch cols"]:
    out_sharding = None
    if not isinstance(indices, int):
        index_sharding = sharding_of(indices)
        if is_sharded(index_sharding):
            out_sharding = NamedSharding(index_sharding.mesh, PartitionSpec(*index_sharding.spec, None))
    return array.at[indices, :].get(out_sharding=out_sharding)
