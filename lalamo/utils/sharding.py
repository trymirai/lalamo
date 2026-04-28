from jax import Array, reshard, typeof
from jax.sharding import NamedSharding, PartitionSpec, get_mesh

from lalamo.module import ShardingAxis

__all__ = [
    "make_sharding",
    "reshard_as",
]


def make_sharding(partition: tuple[ShardingAxis | None, ...] | None) -> NamedSharding | None:
    mesh = get_mesh()
    if partition is None:
        return None
    return NamedSharding(mesh, PartitionSpec(*partition))


def reshard_as(array: Array, reference: Array) -> Array:
    reference_sharding = typeof(reference).sharding
    if not isinstance(reference_sharding, NamedSharding) or reference_sharding.mesh.empty:
        return array
    return reshard(array, reference_sharding)
