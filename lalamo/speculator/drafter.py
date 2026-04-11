from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

from lalamo.speculator.trie import TrieNode

if TYPE_CHECKING:
    from lalamo.speculator.speculate import LMState


class Drafter(ABC):
    """Base class for all drafters. Name-based serialization registry."""
    _registry: ClassVar[dict[str, type[Drafter]]] = {}

    @classmethod
    def register(cls, name: str) -> type[Drafter]:
        def decorator(subcls: type[Drafter]) -> type[Drafter]:
            cls._registry[name] = subcls
            return subcls

        return decorator

    @classmethod
    def deserialize(cls, name: str, data: bytes, **kwargs: object) -> Drafter:
        """Deserialize a drafter by name from a binary blob.

        Requires that the drafter module has been imported (to trigger
        registration). This happens automatically when importing from
        ``lalamo.speculator``.
        """
        subcls = cls._registry.get(name)
        if subcls is None:
            known = ", ".join(cls._registry)
            raise ValueError(f"Unknown drafter {name!r}. Registered: {known}")
        return subcls._deserialize(data, **kwargs)

    @classmethod
    @abstractmethod
    def _deserialize(cls, data: bytes, **kwargs: object) -> Self: ...

    @abstractmethod
    def draft(self, lm: LMState, seed: int) -> TrieNode:
        """Root token must be ``lm.bonus``. Children are continuations after it."""
        ...

    def update_after_verify(
        self,
        prev_lm: LMState,
        accepted: list[int],
        bonus: int,
        new_lm: LMState,
    ) -> Drafter:
        """Post-verify lifecycle hook. Returns updated drafter."""
        return self

    @abstractmethod
    def serialize(self) -> bytes: ...
