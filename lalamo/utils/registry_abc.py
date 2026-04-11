from abc import ABC
from typing import Any, Self
from weakref import WeakSet

__all__ = ["RegistryABC"]


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
