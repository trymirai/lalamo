from abc import ABC
from collections.abc import Callable
from types import UnionType
from typing import Any, Self, get_origin
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


# bootstrap the root
RegistryABC.__registry_descendants__ = WeakSet()


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
