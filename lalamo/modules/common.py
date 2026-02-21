import contextlib
import contextvars
import dataclasses
from abc import abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import Any, Generic, Self, TypeVar

import equinox as eqx
import jax
import jax.sharding as shd
from cattrs import Converter
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterTree, require_array, require_tree

__all__ = [
    "DummyUnionMember",
    "ForwardPassMode",
    "LalamoModule",
    "MeshConfig",
    "ParameterTree",
    "PositionalEmbeddingSelector",
    "Sharding",
    "ShardingOrder",
    "TensorSharding",
    "apply_data_sharding",
    "apply_tensor_sharding",
    "config_converter",
    "get_default_mesh",
    "is_sharded_along",
    "register_config_union",
    "require_array",
    "require_tree",
    "shard_array",
    "sharded_field",
]


class PositionalEmbeddingSelector(Enum):
    GLOBAL = "global"
    LOCAL = "sliding_window"
    NONE = "none"


class ForwardPassMode(Enum):
    MULTI_TOKEN = "multi_token"
    SINGLE_TOKEN = "single_token"


ConfigT_co = TypeVar("ConfigT_co", covariant=True)


class LalamoModule(eqx.Module, Generic[ConfigT_co]):  # noqa: UP046
    config: ConfigT_co = eqx.field(static=True)

    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @abstractmethod
    def export_weights(self) -> ParameterTree[Array]: ...

    @abstractmethod
    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self: ...


def _dtype_to_str(dtype: DTypeLike) -> str:
    if dtype == jnp.bfloat16:
        return "bfloat16"
    try:
        return str(dtype.dtype)  # type: ignore
    except AttributeError:
        return str(dtype)


def _str_to_dtype(dtype_str: str) -> jnp.dtype:
    return {
        "int4": jnp.int4,
        "int8": jnp.int8,
        "int16": jnp.int16,
        "int32": jnp.int32,
        "int64": jnp.int64,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
        "float64": jnp.float64,
    }[dtype_str]


config_converter = Converter()


config_converter.register_unstructure_hook_func(
    lambda t: t in [jnp.dtype, DTypeLike],
    _dtype_to_str,
)

config_converter.register_structure_hook_func(
    lambda t: t in [jnp.dtype, DTypeLike],
    lambda s, _: _str_to_dtype(s),
)


def register_config_union(union_type: UnionType) -> None:
    union_members = union_type.__args__
    name_to_type = {m.__name__: m for m in union_members}

    def unstructure(obj: object) -> dict | None:
        if obj is None:
            return None
        return {
            "type": obj.__class__.__name__,
            **config_converter.unstructure(obj),
        }

    config_converter.register_unstructure_hook(
        union_type,
        unstructure,
    )

    config_converter.register_unstructure_hook(
        union_type | None,
        unstructure,
    )

    def structure[T](config: dict | None, _: type[T]) -> T | None:
        if config is None:
            return None
        new_config = dict(config)
        type_name = new_config.pop("type")
        target_type = name_to_type[type_name]
        return config_converter.structure(new_config, target_type)

    config_converter.register_structure_hook(
        union_type,
        structure,
    )

    config_converter.register_structure_hook(
        union_type | None,
        structure,
    )


@dataclass(frozen=True)
class Sharding:
    tensor_parallelism: int | None = None
    data_parallelism: int | None = None
    data_axis_name: str = "data"
    tensor_axis_name: str = "tensor"

    def resolve(self) -> "MeshConfig":
        device_count = jax.device_count()
        tp = self.tensor_parallelism
        dp = self.data_parallelism

        if (tp is not None and tp <= 0) or (dp is not None and dp <= 0):
            raise ValueError("tensor_parallelism and data_parallelism must be positive integers.")

        if tp is None and dp is None:
            tp, dp = 1, device_count
        elif tp is None:
            assert dp is not None
            if device_count % dp != 0:
                raise ValueError(
                    f"data_parallelism * tensor_parallelism must equal the number of devices ({device_count}).",
                )
            tp = device_count // dp
        elif dp is None:
            if device_count % tp != 0:
                raise ValueError(
                    f"data_parallelism * tensor_parallelism must equal the number of devices ({device_count}).",
                )
            dp = device_count // tp
        elif tp * dp != device_count:
            raise ValueError(
                f"data_parallelism * tensor_parallelism must equal the number of devices ({device_count}).",
            )

        assert tp is not None
        assert dp is not None
        assert tp * dp == device_count

        mesh = jax.make_mesh(
            (dp, tp),
            (self.data_axis_name, self.tensor_axis_name),
            axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
        )
        return MeshConfig(
            data_axis_name=self.data_axis_name,
            tensor_axis_name=self.tensor_axis_name,
            mesh=mesh,
        )


@dataclass(frozen=True)
class MeshConfig:
    data_axis_name: str = "data"
    tensor_axis_name: str = "tensor"
    mesh: shd.Mesh | None = None

    @property
    def data_axis_size(self) -> int:
        if self.mesh is not None:
            return self.mesh.shape[self.data_axis_name]
        abstract_mesh = jax.sharding.get_abstract_mesh()
        if abstract_mesh.empty:
            raise RuntimeError("No mesh provided to MeshConfig and no global mesh set via jax.set_mesh().")
        return abstract_mesh.shape[self.data_axis_name]

    @property
    def tensor_axis_size(self) -> int:
        if self.mesh is not None:
            return self.mesh.shape[self.tensor_axis_name]
        abstract_mesh = jax.sharding.get_abstract_mesh()
        if abstract_mesh.empty:
            raise RuntimeError("No mesh provided to MeshConfig and no global mesh set via jax.set_mesh().")
        return abstract_mesh.shape[self.tensor_axis_name]

    def make_sharding(self, pspec: shd.PartitionSpec) -> shd.NamedSharding | shd.PartitionSpec:
        if self.mesh is not None:
            return shd.NamedSharding(self.mesh, pspec)
        return pspec


_CURRENT_MESH: contextvars.ContextVar[MeshConfig | None] = contextvars.ContextVar(
    "lalamo_current_mesh",
    default=None,
)


@contextlib.contextmanager
def use_mesh(mesh: MeshConfig | None) -> Generator[None, None, None]:
    token = _CURRENT_MESH.set(mesh)
    try:
        yield
    finally:
        _CURRENT_MESH.reset(token)


def get_default_mesh() -> MeshConfig | None:
    mesh = _CURRENT_MESH.get()
    if mesh is not None:
        return mesh
    return None


class ShardingOrder(Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class TensorSharding:
    axes: tuple[int, ...]
    axes_names: tuple[ShardingOrder, ...] | None = None

    def __init__(
        self,
        axes: int | tuple[int, ...],
        axes_names: tuple[ShardingOrder, ...] | None = None,
    ) -> None:
        normalized_axes = (axes,) if isinstance(axes, int) else axes
        if not normalized_axes:
            raise ValueError("TensorSharding.axes must contain at least one axis.")
        if axes_names is not None:
            if len(axes_names) != len(normalized_axes):
                raise ValueError("TensorSharding.axes_names length must match axes length.")
            if len(set(axes_names)) != len(axes_names):
                raise ValueError("TensorSharding.axes_names must not contain duplicates.")
        object.__setattr__(self, "axes", normalized_axes)
        object.__setattr__(self, "axes_names", axes_names)

    def dims_to_try(self, sharding_order: ShardingOrder | None = None) -> tuple[int, ...]:
        if (
            sharding_order is None
            or self.axes_names is None
            or sharding_order not in self.axes_names
            or len(self.axes) == 1
        ):
            return self.axes
        pivot = self.axes_names.index(sharding_order)
        return (self.axes[pivot], *self.axes[:pivot], *self.axes[pivot + 1 :])


def sharded_field(
    tensor_sharding: TensorSharding | None = None,
    min_size_to_shard: int = 10_000,
    **kwargs: object,
) -> Any:  # noqa: ANN401
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["tensor_sharding"] = tensor_sharding
    metadata["min_size_to_shard"] = min_size_to_shard
    return eqx.field(metadata=metadata, **kwargs)


def shard_array(
    array: Array,
    tensor_sharding: TensorSharding,
    mesh: MeshConfig,
    sharding_order: ShardingOrder | None = None,
    min_size_to_shard: int = 10_000,
) -> Array:
    if array.size < min_size_to_shard:
        return array
    axis_size = mesh.tensor_axis_size
    parts: list[str | None] = [None] * array.ndim
    for dim in tensor_sharding.dims_to_try(sharding_order):
        if dim < array.ndim and array.shape[dim] % axis_size == 0:
            parts[dim] = mesh.tensor_axis_name
            break
    pspec = shd.PartitionSpec(*parts)
    return jax.device_put(array, mesh.make_sharding(pspec))


def shard_batch_axis(array: Array, mesh: MeshConfig) -> Array:
    if array.ndim == 0:
        return array
    parts: list[str | None] = [None] * array.ndim
    parts[0] = mesh.data_axis_name
    pspec = shd.PartitionSpec(*parts)
    return jax.lax.with_sharding_constraint(array, mesh.make_sharding(pspec))


def is_sharded_along(
    array: Array,
    *,
    data: bool | None = None,
    tensor: bool | None = None,
    mesh: MeshConfig | None = None,
) -> bool:
    if mesh is None:
        mesh = get_default_mesh()
    if mesh is None:
        return True

    sharding = array.sharding
    if isinstance(sharding, shd.NamedSharding):
        spec = sharding.spec
    elif isinstance(sharding, shd.PartitionSpec):
        spec = sharding
    else:
        return data is not True and tensor is not True

    def has_axis(entry: str | tuple[str, ...] | None, axis_name: str) -> bool:
        if entry is None:
            return False
        if isinstance(entry, str):
            return entry == axis_name
        return axis_name in entry

    if data is not None:
        has_data = len(spec) > 0 and has_axis(spec[0], mesh.data_axis_name)
        if data != has_data:
            return False

    if tensor is not None:
        has_tensor = any(has_axis(entry, mesh.tensor_axis_name) for entry in spec)
        if tensor != has_tensor:
            return False

    return True


def apply_data_sharding[T](value: T, mesh: MeshConfig | None) -> T:
    if mesh is None:
        return value
    return jax.tree_util.tree_map(
        lambda leaf: shard_batch_axis(leaf, mesh) if isinstance(leaf, jax.Array) else leaf,
        value,
    )


def apply_tensor_sharding[T: eqx.Module](
    module: T,
    mesh: MeshConfig | None = None,
    sharding_order: ShardingOrder | None = None,
) -> T:
    mesh = mesh or getattr(module, "mesh", None) or get_default_mesh()
    if mesh is None:
        return module
    sharding_order = sharding_order or getattr(module, "sharding_order", None)
    updates = {
        field.name: shard_array(
            value,
            tensor_sharding,
            mesh,
            sharding_order,
            field.metadata.get("min_size_to_shard", 10_000),
        )
        for field in dataclasses.fields(module)
        if (tensor_sharding := field.metadata.get("tensor_sharding")) is not None
        and isinstance((value := getattr(module, field.name)), jax.Array)
    }
    return dataclasses.replace(module, **updates) if updates else module


@dataclass
class DummyUnionMember:
    def __getattribute__(self, name: str, /) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError
