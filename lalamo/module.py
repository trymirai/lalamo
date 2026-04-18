from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Generic, Self, TypeVar, dataclass_transform

import equinox as eqx
import jax.tree_util as jtu
from cattrs import GenConverter

from lalamo.utils.json import JSON
from lalamo.utils.registry_abc import make_registry_abc_converter

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


@dataclass(frozen=True)
class LalamoConfig:
    _converter: ClassVar[GenConverter] = make_registry_abc_converter()

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
) -> Any:  # noqa: ANN401
    metadata = {"trainable": trainable, "norm": norm, **(metadata or {})}
    return eqx.field(
        metadata=metadata,
        **kwargs,
    )


@dataclass_transform(field_specifiers=(eqx.field, field))
class LalamoModule(Exportable, eqx.Module, Generic[ConfigT_co]):  # noqa: UP046
    config: ConfigT_co = eqx.field(static=True)
