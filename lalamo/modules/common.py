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
    "ParameterTree",
    "PositionalEmbeddingSelector",
    "Sharding",
    "ShardingConfig",
    "apply_data_sharding",
    "apply_sharding",
    "config_converter",
    "get_default_sharding_config",
    "host_creation",
    "init_parallelism",
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

    def resolve(self) -> "ShardingConfig":
        error_message = (
            "Wrong sharding axes: use positive data/tensor parallelism with "
            "data_parallelism * tensor_parallelism == number of devices."
        )
        device_count = jax.device_count()
        tensor_parallelism = self.tensor_parallelism
        data_parallelism = self.data_parallelism
        if tensor_parallelism is not None and tensor_parallelism <= 0:
            raise ValueError(error_message)
        if data_parallelism is not None and data_parallelism <= 0:
            raise ValueError(error_message)
        if tensor_parallelism is None and data_parallelism is None:
            data_parallelism = device_count
            tensor_parallelism = 1
        elif tensor_parallelism is None:
            if data_parallelism is None:
                raise ValueError(error_message)
            if device_count % data_parallelism != 0:
                raise ValueError(error_message)
            tensor_parallelism = device_count // data_parallelism
        elif data_parallelism is None:
            if device_count % tensor_parallelism != 0:
                raise ValueError(error_message)
            data_parallelism = device_count // tensor_parallelism
        elif data_parallelism * tensor_parallelism != device_count:
            raise ValueError(error_message)
        mesh = jax.make_mesh(
            (data_parallelism, tensor_parallelism),
            (self.data_axis_name, self.tensor_axis_name),
            axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
        )
        return ShardingConfig(
            data_axis_name=self.data_axis_name,
            tensor_axis_name=self.tensor_axis_name,
            mesh=mesh,
        )


@dataclass(frozen=True)
class ShardingConfig:
    data_axis_name: str = "data"
    tensor_axis_name: str = "tensor"
    mesh: shd.Mesh | None = None

    @property
    def data_axis_size(self) -> int:
        if self.mesh is not None:
            return self.mesh.shape[self.data_axis_name]
        abstract_mesh = jax.sharding.get_abstract_mesh()
        if abstract_mesh.empty:
            raise RuntimeError("No mesh provided to ShardingConfig and no global mesh set via jax.set_mesh().")
        return abstract_mesh.shape[self.data_axis_name]

    @property
    def tensor_axis_size(self) -> int:
        if self.mesh is not None:
            return self.mesh.shape[self.tensor_axis_name]
        abstract_mesh = jax.sharding.get_abstract_mesh()
        if abstract_mesh.empty:
            raise RuntimeError("No mesh provided to ShardingConfig and no global mesh set via jax.set_mesh().")
        return abstract_mesh.shape[self.tensor_axis_name]

    def _make_sharding(self, pspec: shd.PartitionSpec) -> shd.NamedSharding | shd.PartitionSpec:
        if self.mesh is not None:
            return shd.NamedSharding(self.mesh, pspec)
        return pspec


_CURRENT_SHARDING_CONFIG: contextvars.ContextVar[ShardingConfig | None] = contextvars.ContextVar(
    "lalamo_current_sharding_config",
    default=None,
)


@contextlib.contextmanager
def use_sharding_config(sharding_config: ShardingConfig | None) -> Generator[None, None, None]:
    token = _CURRENT_SHARDING_CONFIG.set(sharding_config)
    try:
        yield
    finally:
        _CURRENT_SHARDING_CONFIG.reset(token)


def get_default_sharding_config() -> ShardingConfig | None:
    sharding_config = _CURRENT_SHARDING_CONFIG.get()
    if sharding_config is not None:
        return sharding_config
    abstract_mesh = jax.sharding.get_abstract_mesh()
    if abstract_mesh.empty:
        return None
    axis_names = tuple(str(name) for name in abstract_mesh.axis_names)
    if "data" in axis_names and "tensor" in axis_names:
        return ShardingConfig()
    if len(axis_names) == 2:
        return ShardingConfig(data_axis_name=axis_names[0], tensor_axis_name=axis_names[1])
    return None


def init_parallelism(
    *,
    tensor_parallelism: int | None = None,
    data_parallelism: int | None = None,
    data_axis_name: str = "data",
    tensor_axis_name: str = "tensor",
) -> None:
    sharding = Sharding(
        tensor_parallelism=tensor_parallelism,
        data_parallelism=data_parallelism,
        data_axis_name=data_axis_name,
        tensor_axis_name=tensor_axis_name,
    )
    resolved = sharding.resolve()
    assert resolved.mesh is not None
    jax.set_mesh(resolved.mesh)


@contextlib.contextmanager
def host_creation() -> Generator[None, None, None]:
    cpu_devices = jax.devices("cpu")
    if cpu_devices:
        with jax.default_device(cpu_devices[0]):
            yield
    else:
        yield


def sharded_field(tensor_dim_order: tuple[int, ...] | None = None, **kwargs: Any) -> Any:  # noqa: ANN401
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["tensor_dim_order"] = tensor_dim_order
    return eqx.field(metadata=metadata, **kwargs)


def shard_array(
    array: Array,
    tensor_dim_order: tuple[int, ...],
    sharding_config: ShardingConfig,
) -> Array:
    axis_size = sharding_config.tensor_axis_size
    parts: list[str | None] = [None] * array.ndim
    for dim in tensor_dim_order:
        if dim < array.ndim and array.shape[dim] % axis_size == 0:
            parts[dim] = sharding_config.tensor_axis_name
            break
    pspec = shd.PartitionSpec(*parts)
    return jax.device_put(array, sharding_config._make_sharding(pspec))


def shard_batch_axis(array: Array, sharding_config: ShardingConfig) -> Array:
    if array.ndim == 0:
        return array
    parts: list[str | None] = [None] * array.ndim
    parts[0] = sharding_config.data_axis_name
    pspec = shd.PartitionSpec(*parts)
    return jax.lax.with_sharding_constraint(array, sharding_config._make_sharding(pspec))


def apply_data_sharding[T](value: T, sharding_config: ShardingConfig | None) -> T:
    if sharding_config is None:
        return value
    return jax.tree_util.tree_map(
        lambda leaf: shard_batch_axis(leaf, sharding_config) if isinstance(leaf, jax.Array) else leaf,
        value,
    )


def apply_sharding[T: eqx.Module](module: T, sharding_config: ShardingConfig | None) -> T:
    if sharding_config is None:
        return module
    updates: dict[str, Array] = {}
    for field in dataclasses.fields(module):
        tensor_dim_order = field.metadata.get("tensor_dim_order")
        if tensor_dim_order is None:
            continue
        value = getattr(module, field.name)
        if isinstance(value, jax.Array):
            updates[field.name] = shard_array(value, tensor_dim_order, sharding_config)
    if updates:
        return dataclasses.replace(module, **updates)
    return module


@dataclass
class DummyUnionMember:
    def __getattribute__(self, name: str, /) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError
