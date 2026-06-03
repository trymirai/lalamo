from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeGuard

import jax
from jax import Array, ShapeDtypeStruct, typeof
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from jaxtyping import Int, Shaped

__all__ = [
    "LogicalAxis",
    "ShardingConfig",
    "is_sharded",
    "lookup_sharded_indices",
    "reshard_as",
    "sharding_of",
    "with_sharding",
]


class LogicalAxis(StrEnum):
    BATCH = "batch"
    SEQUENCE = "sequence"
    MATRIX = "matrix"
    MIXTURE = "mixture"


@dataclass(frozen=True)
class ShardingConfig:
    mesh: Mesh
    logical_to_physical: Mapping[LogicalAxis, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mesh.empty:
            raise ValueError("ShardingConfig requires a non-empty mesh.")

    @staticmethod
    def _mesh(axis_name: str, devices: Sequence[jax.Device] | None = None) -> Mesh:
        if devices is None:
            devices = jax.devices()
            first_device, *_ = devices
            if first_device.platform == "cpu":
                devices = devices[:1]
        return jax.make_mesh(
            (len(devices),),
            (axis_name,),
            axis_types=(AxisType.Explicit,),
            devices=devices,
        )

    @classmethod
    def replicated(cls, devices: Sequence[jax.Device] | None = None) -> "ShardingConfig":
        return cls(mesh=cls._mesh("replica", devices))

    @classmethod
    def data_parallel(cls, devices: Sequence[jax.Device] | None = None) -> "ShardingConfig":
        return cls(
            mesh=cls._mesh("data", devices),
            logical_to_physical={LogicalAxis.BATCH: "data"},
        )

    @classmethod
    def expert_parallel(cls, devices: Sequence[jax.Device] | None = None) -> "ShardingConfig":
        return cls(
            mesh=cls._mesh("expert", devices),
            logical_to_physical={LogicalAxis.MIXTURE: "expert"},
        )

    @classmethod
    def fully_sharded_data_parallel(cls, devices: Sequence[jax.Device] | None = None) -> "ShardingConfig":
        return cls(
            mesh=cls._mesh("fsdp", devices),
            logical_to_physical={
                LogicalAxis.BATCH: "fsdp",
                LogicalAxis.MATRIX: "fsdp",
                LogicalAxis.MIXTURE: "fsdp",
            },
        )

    def resolve_axis(self, logical_axis: LogicalAxis | None) -> str | None:
        if logical_axis is None:
            return None
        return self.logical_to_physical.get(logical_axis)

    def resolve_sharding(self, logical_axes: Iterable[LogicalAxis | None]) -> NamedSharding:
        physical_axes = tuple(self.resolve_axis(logical_axis) for logical_axis in logical_axes)
        return self.make_sharding(physical_axes)

    def make_sharding(self, physical_axes: Iterable[str | None]) -> NamedSharding:
        physical_axes = tuple(physical_axes)
        sharded_axes = tuple(axis for axis in physical_axes if axis is not None)
        if len(set(sharded_axes)) != len(sharded_axes):
            raise ValueError(f"Cannot shard multiple array axes over the same mesh axis: {physical_axes}")
        return NamedSharding(self.mesh, PartitionSpec(*physical_axes))


def is_sharded(sharding: object) -> TypeGuard[NamedSharding]:
    if not isinstance(sharding, NamedSharding) or sharding.mesh.empty:
        return False
    return any(axis is not None for axis in sharding.spec)


def sharding_of(array: Array | ShapeDtypeStruct) -> NamedSharding:
    try:
        sharding = array.sharding
    except AttributeError:
        sharding = typeof(array).sharding
    assert isinstance(sharding, NamedSharding)
    return sharding


def with_sharding(array: Array, sharding: NamedSharding) -> Array:
    if isinstance(array, ShapeDtypeStruct):
        from lalamo.utils.dummy_array import dummy_array  # noqa: PLC0415

        return dummy_array(array.shape, array.dtype, sharding, weak_type=array.weak_type)
    return jax.device_put(array, sharding)


def reshard_as(array: Array, reference: Array) -> Array:
    return with_sharding(array, sharding_of(reference))


def lookup_sharded_indices(
    array: Shaped[Array, "rows cols"],
    row_index: int | Int[Array, "*batch"],
    out_sharding: NamedSharding | None = None,
) -> Shaped[Array, "*batch cols"]:
    if out_sharding is None:
        if isinstance(row_index, int):
            array_sharding = sharding_of(array)
            out_sharding = NamedSharding(array_sharding.mesh, PartitionSpec(None))
        else:
            row_index_sharding = sharding_of(row_index)
            out_sharding = NamedSharding(row_index_sharding.mesh, PartitionSpec(*row_index_sharding.spec, None))
    result = array.at[row_index, :].get(out_sharding=out_sharding)
    if isinstance(out_sharding.mesh, Mesh):
        return jax.device_put(result, out_sharding)
    return result
