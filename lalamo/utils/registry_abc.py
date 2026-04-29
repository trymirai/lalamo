from abc import ABC
from collections.abc import Callable
from types import UnionType
from typing import Any, Self, TypeVar, Union, cast, get_args, get_origin
from weakref import WeakSet

import jax.numpy as jnp
from cattrs import GenConverter
from cattrs.preconf.json import make_converter
from jaxtyping import DTypeLike

from lalamo.utils.dtype import dtype_to_str, str_to_dtype

__all__ = [
    "RegistryABC",
    "make_registry_abc_converter",
]


class RegistryABC(ABC):
    """
    Abstract base that tracks descendants via __init_subclass__.

    Any class defined as `class AbstractFoo(RegistryABC)` will expose a
    class method `AbstractFoo.__descendants__()` that returns a tuple of
    all concrete classes having AbstractFoo in their MRO, excluding classes
    that directly list `RegistryABC` among their bases.
    """

    __registry_descendants__: WeakSet[type["RegistryABC"]]

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        cls.__registry_descendants__ = WeakSet()

        for ancestor in cls.__mro__[1:]:
            if (
                ancestor is not RegistryABC
                and issubclass(ancestor, RegistryABC)
                and not any(b is RegistryABC for b in cls.__bases__)
            ):
                ancestor.__registry_descendants__.add(cls)

    @classmethod
    def __descendants__(cls) -> tuple[type[Self], ...]:
        return tuple(cls.__registry_descendants__)


RegistryABC.__registry_descendants__ = WeakSet()


def _strip_none_from_optional(annotation: object) -> object | None:
    origin = get_origin(annotation)
    if origin is not Union and not isinstance(annotation, UnionType):
        return annotation

    non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(non_none_args) != 1:
        return None
    return non_none_args[0]


def _typevar_bound(annotation: object) -> object:
    if isinstance(annotation, TypeVar):
        return annotation.__bound__
    return annotation


def _maybe_registry_abc_from_option(annotation: object) -> type[RegistryABC] | None:
    maybe_registry_abc = _typevar_bound(_strip_none_from_optional(annotation))

    if not isinstance(maybe_registry_abc, type) or not issubclass(maybe_registry_abc, RegistryABC):
        return None

    return maybe_registry_abc


def _maybe_registry_abc_from_type_option(annotation: object) -> type[RegistryABC] | None:
    maybe_type_annotation = _strip_none_from_optional(annotation)
    if get_origin(maybe_type_annotation) is not type:
        return None

    args = get_args(maybe_type_annotation)
    if len(args) != 1:
        return None

    maybe_registry_abc = args[0]
    if not isinstance(maybe_registry_abc, type) or not issubclass(maybe_registry_abc, RegistryABC):
        return None

    return maybe_registry_abc


def _descendant_by_name(registry_abc: type[RegistryABC], type_name: str) -> type[RegistryABC]:
    if registry_abc.__name__ == type_name:
        return registry_abc

    name_to_descendant = {descendant.__name__: descendant for descendant in registry_abc.__descendants__()}
    try:
        return name_to_descendant[type_name]
    except KeyError as e:
        available = ", ".join(sorted(name_to_descendant))
        raise ValueError(f"Unknown {registry_abc.__name__} descendant: {type_name!r}. Available: {available}") from e


def make_registry_abc_converter() -> GenConverter:
    converter = make_converter()

    converter.register_unstructure_hook_func(
        lambda t: t in [jnp.dtype, DTypeLike],
        dtype_to_str,
    )
    converter.register_structure_hook_func(
        lambda t: t in [jnp.dtype, DTypeLike],
        lambda s, _: str_to_dtype(s),
    )

    def is_registry_abc(maybe_registry_abc: object) -> bool:
        return _maybe_registry_abc_from_option(maybe_registry_abc) is not None

    def is_registry_abc_type(maybe_registry_abc_type: object) -> bool:
        return _maybe_registry_abc_from_type_option(maybe_registry_abc_type) is not None

    @converter.register_unstructure_hook_factory(is_registry_abc_type)
    def unstructure_abc_type_factory(
        maybe_registry_abc_type: type[type[RegistryABC]] | type[type[RegistryABC] | None],
    ) -> Callable[[object], str | None]:
        registry_abc = _maybe_registry_abc_from_type_option(maybe_registry_abc_type)
        if registry_abc is None:
            raise TypeError(f"Expected a RegistryABC type annotation, got {maybe_registry_abc_type}")
        resolved_registry_abc: type[RegistryABC] = registry_abc

        def unstructure_abc_type(obj: object) -> str | None:
            if obj is None:
                return None
            if not isinstance(obj, type) or not issubclass(obj, resolved_registry_abc):
                raise TypeError(f"Expected a {resolved_registry_abc.__name__} subclass, got {obj}")
            return obj.__name__

        return unstructure_abc_type

    @converter.register_structure_hook_factory(is_registry_abc_type)
    def structure_abc_type_factory(
        maybe_registry_abc_type: type[type[RegistryABC]] | type[type[RegistryABC] | None],
    ) -> Callable[[str | type[RegistryABC] | None, type[type[RegistryABC]]], type[RegistryABC] | None]:
        registry_abc = _maybe_registry_abc_from_type_option(maybe_registry_abc_type)
        if registry_abc is None:
            raise TypeError(f"Expected a RegistryABC type annotation, got {maybe_registry_abc_type}")
        resolved_registry_abc: type[RegistryABC] = registry_abc

        def structure_abc_type(
            value: str | type[RegistryABC] | None,
            _: type[type[RegistryABC]],
        ) -> type[RegistryABC] | None:
            if value is None:
                return None
            if isinstance(value, type) and issubclass(value, resolved_registry_abc):
                return value
            return _descendant_by_name(resolved_registry_abc, cast("str", value))

        return structure_abc_type

    @converter.register_unstructure_hook_factory(is_registry_abc)
    def unstructure_abc_factory(
        _: type[RegistryABC] | type[RegistryABC | None],
        converter: GenConverter,
    ) -> Callable[[object], dict[str, Any] | None]:
        def unstructure_abc(obj: object) -> dict[str, Any] | None:
            if obj is None:
                return None
            return {
                "type": obj.__class__.__name__,
                **converter.unstructure_attrs_asdict(obj),
            }

        return unstructure_abc

    @converter.register_structure_hook_factory(is_registry_abc)
    def structure_abc_factory(
        maybe_registry_abc: type[RegistryABC] | type[RegistryABC | None],
    ) -> Callable[[dict[str, Any] | None, type[RegistryABC] | type[RegistryABC | None]], RegistryABC | None]:
        registry_abc = _maybe_registry_abc_from_option(maybe_registry_abc)
        if registry_abc is None:
            raise TypeError(f"Expected a RegistryABC subclass, got {maybe_registry_abc}")
        resolved_registry_abc: type[RegistryABC] = registry_abc

        def structure_abc(
            config: dict[str, Any] | None,
            _: type[RegistryABC] | type[RegistryABC | None],
        ) -> RegistryABC | None:
            if config is None:
                return None

            new_config = dict(config)
            type_name = cast("str", new_config.pop("type"))
            target_type = _descendant_by_name(resolved_registry_abc, type_name)
            return converter.structure_attrs_fromdict(new_config, target_type)

        return structure_abc

    return converter
