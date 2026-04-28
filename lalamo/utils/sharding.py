from jax.sharding import NamedSharding, PartitionSpec, get_mesh

from lalamo.module import ShardingAxis

__all__ = [
    "make_sharding",
]


def make_sharding(partition: tuple[ShardingAxis | None, ...] | None) -> NamedSharding | None:
    mesh = get_mesh()
    if partition is None:
        return None
    return NamedSharding(mesh, PartitionSpec(*partition))
