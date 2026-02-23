import contextlib
import contextvars
import dataclasses
from abc import abstractmethod
from collections.abc import Callable, Generator, Mapping
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
    "apply_tensor_sharding_recursive",
    "config_converter",
    "get_default_mesh",
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
    fsdp: bool = False
    data_axis_name: str = "data"
    tensor_axis_name: str = "tensor"

    def resolve(self) -> "MeshConfig":
        device_count = jax.device_count()
        tp = self.tensor_parallelism
        dp = self.data_parallelism

        if (tp is not None and tp <= 0) or (dp is not None and dp <= 0):
            raise ValueError("tensor_parallelism and data_parallelism must be positive integers.")

        match (tp, dp):
            case (tp, None):
                dp = device_count // tp
            case (None, dp):
                tp = device_count // dp

        if tp * dp != device_count:  # type: ignore
            raise ValueError(
                f"data_parallelism * tensor_parallelism must equal the number of devices ({device_count}).",
            )

        mesh = jax.make_mesh(
            (dp, tp),  # type: ignore
            (self.data_axis_name, self.tensor_axis_name),
            axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
        )
        return MeshConfig(
            data_axis_name=self.data_axis_name,
            tensor_axis_name=self.tensor_axis_name,
            mesh=mesh,
            fsdp=self.fsdp,
        )


@dataclass(frozen=True)
class MeshConfig:
    data_axis_name: str = "data"
    tensor_axis_name: str = "tensor"
    mesh: shd.Mesh | None = None
    fsdp: bool = False

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
    return _CURRENT_MESH.get()


class ShardingOrder(Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class TensorSharding:
    axes: tuple[int, ...]
    axes_names: tuple[ShardingOrder, ...] | None = None

    @classmethod
    def build(
        cls,
        axes: int | tuple[int, ...],
        axes_names: tuple[ShardingOrder, ...] | None = None,
    ) -> "TensorSharding":
        normalized_axes = (axes,) if isinstance(axes, int) else axes
        return TensorSharding(normalized_axes, axes_names)

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("TensorSharding.axes must contain at least one axis.")

        if self.axes_names is not None:
            if len(self.axes_names) != len(self.axes):
                raise ValueError("TensorSharding.axes_names length must match axes length.")
            if len(set(self.axes_names)) != len(self.axes_names):
                raise ValueError("TensorSharding.axes_names must not contain duplicates.")

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
    min_size_to_shard: int = 2**18,
    *,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    merged_metadata = dict(metadata or {})
    merged_metadata["tensor_sharding"] = tensor_sharding
    merged_metadata["min_size_to_shard"] = min_size_to_shard
    return eqx.field(
        converter=converter,
        static=static,
        metadata=merged_metadata,
        **kwargs,
    )


def shard_array(
    array: Array,
    tensor_sharding: TensorSharding,
    mesh: MeshConfig,
    sharding_order: ShardingOrder | None = None,
    min_size_to_shard: int = 2**18,
) -> Array:
    if array.size < min_size_to_shard:
        return array
    parts: list[str | None] = [None] * array.ndim

    tp_dim: int | None = None
    tp_axis_size = mesh.tensor_axis_size
    for dim in tensor_sharding.dims_to_try(sharding_order):
        if dim < array.ndim and array.shape[dim] % tp_axis_size == 0:
            parts[dim] = mesh.tensor_axis_name
            tp_dim = dim
            break

    if mesh.fsdp:
        dp_axis_size = mesh.data_axis_size
        for dim in tensor_sharding.axes:
            if dim != tp_dim and dim < array.ndim and array.shape[dim] % dp_axis_size == 0:
                parts[dim] = mesh.data_axis_name
                break

    pspec = shd.PartitionSpec(*parts)
    return jax.device_put(array, mesh.make_sharding(pspec))


def shard_batch_axis(array: Array, mesh: MeshConfig, *, batch_axis: int) -> Array:
    if array.ndim == 0:
        return array
    parts: list[str | None] = [None] * array.ndim
    parts[batch_axis] = mesh.data_axis_name
    pspec = shd.PartitionSpec(*parts)
    return jax.lax.with_sharding_constraint(array, mesh.make_sharding(pspec))


def apply_data_sharding[T](value: T, mesh: MeshConfig | None, *, batch_axis: int) -> T:
    if mesh is None:
        return value
    return jax.tree_util.tree_map(
        lambda leaf: shard_batch_axis(leaf, mesh, batch_axis=batch_axis) if eqx.is_array(leaf) else leaf,
        value,
    )


def apply_tensor_sharding[T: eqx.Module](module: T) -> T:
    mesh = get_default_mesh()
    if mesh is None:
        return module
    return apply_tensor_sharding_recursive(module, mesh)


def apply_tensor_sharding_recursive[T: eqx.Module](module: T, mesh: MeshConfig) -> T:
    from .linear import LinearBase

    replacements: dict[int, Array] = {}

    def collect(mod: eqx.Module) -> None:
        sharding_order = mod.sharding_order if isinstance(mod, LinearBase) else None
        for field in dataclasses.fields(mod):
            value = getattr(mod, field.name)
            tensor_sharding = field.metadata.get("tensor_sharding")
            if tensor_sharding is not None and eqx.is_array(value):
                replacements[id(value)] = shard_array(
                    value,
                    tensor_sharding,
                    mesh,
                    sharding_order,
                    field.metadata.get("min_size_to_shard", 0),
                )
            elif isinstance(value, eqx.Module):
                collect(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, eqx.Module):
                        collect(item)

    collect(module)
    if not replacements:
        return module
    return jax.tree.map(lambda leaf: replacements.get(id(leaf), leaf), module)


@dataclass
class DummyUnionMember:
    def __getattribute__(self, name: str, /) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError
