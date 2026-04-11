from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import Field, dataclass
from enum import StrEnum
from types import UnionType
from typing import Any, ClassVar, Generic, Self, TypeVar, dataclass_transform, get_origin

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from cattrs import GenConverter
from cattrs.preconf.json import make_converter
from jaxtyping import DTypeLike

from lalamo.utils.dtype import dtype_to_str, str_to_dtype
from lalamo.utils.registry_abc import RegistryABC

from .exportable import Exportable

__all__ = [
    "ForwardPassMode",
    "LalamoConfig",
    "LalamoModule",
    "ShardingAxis",
    "field",
]


class ForwardPassMode(StrEnum):
    MULTI_TOKEN = "multi_token"
    SINGLE_TOKEN = "single_token"


class ShardingAxis(StrEnum):
    DATA = "data"
    TENSOR = "tensor"
    EXPERT = "expert"


type JSON = str | int | float | bool | None | dict[str, JSON] | list[JSON]


def _make_config_converter() -> GenConverter:
    converter = make_converter()

    converter.register_unstructure_hook_func(
        lambda t: t in [jnp.dtype, DTypeLike],
        dtype_to_str,
    )
    converter.register_structure_hook_func(
        lambda t: t in [jnp.dtype, DTypeLike],
        lambda s, _: str_to_dtype(s),
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


@dataclass(frozen=True)
class LalamoConfig:
    _converter: ClassVar[GenConverter] = _make_config_converter()

    @classmethod
    def from_json(cls, json_object: JSON) -> Self:
        return cls._converter.structure(json_object, cls)

    def to_json(self) -> JSON:
        return self._converter.unstructure(self)


ConfigT_co = TypeVar("ConfigT_co", bound=LalamoConfig, covariant=True)

type PytreeKey = (
    str | int | jtu.GetAttrKey | jtu.SequenceKey | jtu.DictKey | jtu.FlattenedIndexKey | tuple[PytreeKey, ...]
)


class ParameterNorm(StrEnum):
    L_INF = "l_inf"
    L_2 = "l_2"
    SPECTRAL = "spectral"


@dataclass(frozen=True)
class FieldMetadata:
    trainable: bool = True
    norm: ParameterNorm = ParameterNorm.L_INF


def field(
    *,
    trainable: bool = True,
    norm: ParameterNorm = ParameterNorm.L_INF,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Field:
    metadata = {"trainable": trainable, "norm": norm, **(metadata or {})}
    return eqx.field(
        metadata=metadata,
        **kwargs,
    )


@dataclass_transform(frozen_default=True, field_specifiers=(eqx.field, field))
class LalamoModule(Exportable, eqx.Module, Generic[ConfigT_co]):  # noqa: UP046
    config: ConfigT_co = eqx.field(static=True)

    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...


def _unpack_registry_abc_from_option(annotation: type) -> type[RegistryABC]:
    maybe_registry_abc = get_origin(annotation)

    if isinstance(maybe_registry_abc, UnionType):
        maybe_registry_abc, *rest = set(maybe_registry_abc.__args__) - {None}
        if rest:
            raise TypeError(f"Expected a RegistryABC subclass or None, got {annotation}")

    assert maybe_registry_abc is not None
    if not issubclass(maybe_registry_abc, RegistryABC):
        raise TypeError(f"Expected a RegistryABC subclass, got {maybe_registry_abc}")

    return maybe_registry_abc
