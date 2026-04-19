from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar

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


def _field_metadata(
    *,
    trainable: bool | None,
    static: bool,
    norm: ParameterNorm,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if trainable is None:
        trainable = not static
    if static and trainable:
        raise ValueError("A field cannot be both static and trainable.")
    return {"trainable": trainable, "norm": norm, **(metadata or {})}


if TYPE_CHECKING:
    # Pyrefly currently mis-handles wrappers around `eqx.field(...)` when they are
    # inherited through an `eqx.Module` hierarchy. In practice that means a field
    # like `config: ConfigT = field(static=True)` is treated as if it were an
    # ordinary default-valued class attribute rather than a dataclass field, and
    # subclasses then fail with false positives such as:
    #
    #   "Dataclass field x without a default may not follow dataclass field with a default"
    #
    # The failure does not come from our runtime behavior. `eqx.Module` already
    # provides the dataclass transform via its metaclass, and the exact same class
    # hierarchy type-checks correctly when the field helper is written as
    # `eqx.field(...)` directly. Adding another `@dataclass_transform(...)` on
    # `LalamoModule`, or even putting one on a custom metaclass derived from the
    # Equinox metaclass, does not help: pyrefly still fails to recognize the
    # wrapper once the class goes through the `eqx.Module` path. The bug is
    # specifically in the checker's handling of a wrapper around `eqx.field`, even
    # if that wrapper is otherwise a faithful field specifier.
    #
    # So for static analysis we expose `lalamo.field` as a plain alias to
    # `eqx.field`, which lets pyrefly recognize inherited Lalamo fields correctly.
    # At runtime we still use the Lalamo wrapper below, because we need to
    # translate the Lalamo-only `trainable` and `norm` arguments into Equinox
    # metadata before calling `eqx.field(...)`.
    field = eqx.field
else:

    def field(
        *,
        trainable: bool | None = None,
        static: bool = False,
        norm: ParameterNorm = ParameterNorm.L_INF,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        return eqx.field(
            metadata=_field_metadata(trainable=trainable, static=static, norm=norm, metadata=metadata),
            static=static,
            **kwargs,
        )


class LalamoModule(Exportable, eqx.Module, Generic[ConfigT_co]):  # noqa: UP046
    config: ConfigT_co = field(static=True)
