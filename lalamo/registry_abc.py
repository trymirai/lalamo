from abc import ABC, ABCMeta
from typing import Any, Self
from weakref import WeakSet

__all__ = ["RegistryABC", "RegistryMeta"]


class RegistryMeta(ABCMeta):
    """
    Metaclass that tracks, for each subclass of RegistryABC, a per-class WeakSet
    of descendants (classes that have it in their MRO) while excluding any class
    that directly lists `RegistryABC` among its bases.
    """

    _REG_ATTR: str = "__registry_descendants__"
    _ROOT: type["RegistryABC"] | None = None

    def __init__(
        cls: type,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        # Give *every* class its own WeakSet (shadow any inherited attribute)
        setattr(cls, RegistryMeta._REG_ATTR, WeakSet())

        # Detect and remember the root exactly once
        if RegistryMeta._ROOT is None and name == "RegistryABC":
            RegistryMeta._ROOT = cls
            return

        root = RegistryMeta._ROOT
        if root is None:
            # Extremely early import edge-case; nothing to register yet
            return

        # Exclude classes that directly list the root among bases
        if any(b is root for b in cls.__bases__):
            return

        # Register this class on all qualifying ancestors below the root
        for ancestor in cls.__mro__[1:]:
            if isinstance(ancestor, RegistryMeta) and issubclass(ancestor, root):
                getattr(ancestor, RegistryMeta._REG_ATTR).add(cls)


class RegistryABC(ABC, metaclass=RegistryMeta):
    """
    Abstract base tracked by RegistryMeta.

    Any class defined as `class AbstractFoo(RegistryABC)` will expose a
    class method `AbstractFoo.__get_descendants__()` that returns a list of
    all classes having AbstractFoo in their MRO *except* those that directly
    include `RegistryABC` among their bases.
    """

    @classmethod
    def __descendants__(cls) -> tuple[type[Self], ...]:
        reg: WeakSet[type[Self]] = getattr(cls, RegistryMeta._REG_ATTR)  # noqa: SLF001
        return tuple(reg)
