from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import Any, Generic, Self, TypeVar, cast

import equinox as eqx
import jax
from cattrs import Converter
from jax import ShapeDtypeStruct
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Key

from lalamo.common import ParameterPath, ParameterTree
from lalamo.field import FieldMetadata, field, field_metadata_from_path
from lalamo.serialization import UzuSerializable
from lalamo.sharding import (
    ShardingConfig,
    ShardingOrder,
    TensorSharding,
    apply_parameter_sharding,
    get_current_sharding_config,
    pad_and_apply_data_sharding,
    use_sharding,
)

__all__ = [
    "DummyUnionMember",
    "EmptyInitializer",
    "FieldMetadata",
    "ForwardPassMode",
    "Initializer",
    "LalamoModule",
    "ParameterTree",
    "PositionalEmbeddingSelector",
    "RandomInitializer",
    "ShardingConfig",
    "ShardingOrder",
    "TensorSharding",
    "apply_parameter_sharding",
    "config_converter",
    "field",
    "field_metadata_from_path",
    "get_current_sharding_config",
    "pad_and_apply_data_sharding",
    "register_config_union",
    "use_sharding",
]


class ForwardPassMode(Enum):
    MULTI_TOKEN = "multi_token"
    SINGLE_TOKEN = "single_token"


ConfigT_co = TypeVar("ConfigT_co", covariant=True)


class LalamoModule(UzuSerializable, eqx.Module, Generic[ConfigT_co]):  # noqa: UP046
    config: ConfigT_co = eqx.field(static=True)

    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    def shard(self, sharding_config: ShardingConfig) -> Self:
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: apply_parameter_sharding(
                leaf, field_metadata_from_path(self, ParameterPath("") / path), sharding_config
            )
            if eqx.is_array(leaf)
            else leaf,
            self,
        )


@dataclass
class Initializer(ABC):
    precision: DTypeLike

    @abstractmethod
    def normal(self, std: float, shape: tuple[int, ...], dtype: DTypeLike) -> Array: ...

    @abstractmethod
    def ones(self, shape: tuple[int, ...], dtype: DTypeLike) -> Array: ...

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: DTypeLike) -> Array: ...


class EmptyInitializer(Initializer):
    @classmethod
    def _dummy_array(cls, shape: int | tuple[int, ...], dtype: DTypeLike) -> Array:
        if isinstance(shape, int):
            shape = (shape,)
        return cast("Array", ShapeDtypeStruct(shape=shape, dtype=dtype))

    def normal(self, std: float, shape: tuple[int, ...], dtype: DTypeLike) -> Array:  # noqa: ARG002
        return self._dummy_array(shape, dtype)

    def ones(self, shape: tuple[int, ...], dtype: DTypeLike) -> Array:
        return self._dummy_array(shape, dtype)

    def zeros(self, shape: tuple[int, ...], dtype: DTypeLike) -> Array:
        return self._dummy_array(shape, dtype)


@dataclass
class RandomInitializer(Initializer):
    key: Key[Array, ""]

    def normal(self, std: float, shape: tuple[int, ...], dtype: DTypeLike) -> Array:
        self.key, key = jax.random.split(self.key)
        return jax.random.normal(key, shape, dtype) * std

    def ones(self, shape: tuple[int, ...], dtype: DTypeLike) -> Array:
        return jnp.ones(shape, dtype)

    def zeros(self, shape: tuple[int, ...], dtype: DTypeLike) -> Array:
        return jnp.zeros(shape, dtype)


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


@dataclass
class DummyUnionMember:
    def __getattribute__(self, name: str, /) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError
