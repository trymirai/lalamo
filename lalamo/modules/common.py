from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import Any, Generic, Self, TypeVar

import equinox as eqx
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
    "config_converter",
    "register_config_union",
    "require_array",
    "require_tree",
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


@dataclass
class DummyUnionMember:
    def __getattribute__(self, name: str, /) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError
