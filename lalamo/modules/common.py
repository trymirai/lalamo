from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import ClassVar, Generic, Self, TypeVar, get_origin

import equinox as eqx
from cattrs import GenConverter
from cattrs.preconf.json import make_converter
from jax import numpy as jnp
from jaxtyping import DTypeLike

from lalamo.common import JSON, RegistryABC

__all__ = [
    "ForwardPassMode",
    "LalamoModule",
    "PositionalEmbeddingSelector",
]


def _make_config_converter() -> GenConverter:
    converter = make_converter()

    converter.register_unstructure_hook_func(
        lambda t: t in [jnp.dtype, DTypeLike],
        _dtype_to_str,
    )
    converter.register_structure_hook_func(
        lambda t: t in [jnp.dtype, DTypeLike],
        lambda s, _: _str_to_dtype(s),
    )

    def is_registry_abc(maybe_registry_abc: type) -> bool:
        return _unpack_registry_abc_from_option(maybe_registry_abc) is not None

    @converter.register_unstructure_hook_factory(is_registry_abc)
    def unstructure_abc_factory(
        _: type[RegistryABC] | type[RegistryABC | None],
        converter: GenConverter,
    ) -> Callable[[object], dict | None]:
        def unstructure_abc(obj: object) -> dict | None:
            if obj is None:
                return None
            return {
                "type": obj.__class__.__name__,
                **converter.unstructure(obj),
            }

        return unstructure_abc

    @converter.register_structure_hook_factory(is_registry_abc)
    def structure_abc_factory(
        maybe_registry_abc: type[RegistryABC] | type[RegistryABC | None],
    ) -> Callable[[dict | None, type[RegistryABC] | type[RegistryABC | None]], RegistryABC | None]:
        registry_abc = _unpack_registry_abc_from_option(maybe_registry_abc)
        assert registry_abc is not None

        def structure_abc(
            config: dict | None,
            _: type[RegistryABC] | type[RegistryABC | None],
        ) -> RegistryABC | None:
            if config is None:
                return None

            new_config = dict(config)
            type_name = new_config.pop("type")
            target_type = {d.__name__: d for d in registry_abc.__descendants__()}[type_name]
            return converter.structure(new_config, target_type)

        return structure_abc

    return converter


class PositionalEmbeddingSelector(Enum):
    GLOBAL = "global"
    LOCAL = "sliding_window"
    NONE = "none"


class ForwardPassMode(Enum):
    MULTI_TOKEN = "multi_token"
    SINGLE_TOKEN = "single_token"


@dataclass(frozen=True)
class LalamoConfig:
    _converter: ClassVar[GenConverter] = _make_config_converter()

    @classmethod
    def from_json(cls, json_object: JSON) -> Self:
        return cls._converter.structure(json_object, cls)

    def to_json(self) -> JSON:
        return self._converter.unstructure(self)


ConfigT_co = TypeVar("ConfigT_co", bound=LalamoConfig, covariant=True)


class LalamoModule(eqx.Module, Generic[ConfigT_co]):  # noqa: UP046
    config: ConfigT_co = eqx.field(static=True)

    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...


def _unpack_registry_abc_from_option(annotation: type) -> type[RegistryABC] | None:
    maybe_registry_abc = get_origin(annotation)

    if isinstance(maybe_registry_abc, UnionType):
        maybe_registry_abc, *rest = set(maybe_registry_abc.__args__) - {None}
        if rest:
            return None

    assert maybe_registry_abc is not None
    if issubclass(maybe_registry_abc, RegistryABC):
        return maybe_registry_abc

    return None


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
