from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

from typer import Typer

from lalamo.speculator.trie import TrieNode

if TYPE_CHECKING:
    from lalamo.speculator.speculate import LMState


class Drafter(ABC):
    """Base class for all drafters. Name-based serialization registry.

    Drafters declare what activations the speculation runtime should retain in
    ``LMState`` via three attrs:

    - ``trace_layer_outputs``:
        - ``None``     : do not retain any per-layer residual outputs.
        - ``(l, ...)`` : retain the listed layers' residual outputs. Each
          becomes a ``(suffix, channels)`` array in ``LMState.layer_outputs``.
    - ``trace_output_norm``:
        - ``False``: ``LMState.output_norm`` is ``None``.
        - ``True`` : retain the final ``output_norm`` activation (lm_head input).
    - ``prefill_hidden_range``:
        - ``None`` : retain hiddens for every prompt position (full prompt).
        - ``int N``: retain hiddens only for the last N prompt positions.

    These are defaults on the base (fallback: retain nothing). Subclasses may
    override them as class attributes or as dataclass fields — the latter
    when the value is instance-specific (e.g. DFlash's ``trace_layer_outputs``
    depends on the target model's layer count).

    After every verify step the retained slice is ``kept_fwd[:total_kept]``
    (bonus + accepted) regardless of these attrs.

    Lifecycle:
    - ``draft(lm, seed) -> TrieNode``
    - ``on_prefill(lm)``: invoked once after prefill, before the first draft.
    - ``update_after_verify(prev_lm, accepted, bonus, new_lm)``: invoked after
      each verify step.
    """

    trace_layer_outputs: tuple[int, ...] | None = None
    trace_output_norm: bool = False
    prefill_hidden_range: int | None = None

    _registry: ClassVar[dict[str, "type[Drafter]"]] = {}

    @classmethod
    def register(cls, name: str) -> "type[Drafter]":
        def decorator(subcls: "type[Drafter]") -> "type[Drafter]":
            cls._registry[name] = subcls
            return subcls

        return decorator

    @classmethod
    def deserialize(cls, name: str, data: bytes, **kwargs: object) -> "Drafter":
        """Deserialize a drafter by name from a binary blob.

        Requires that the drafter module has been imported (to trigger
        registration). This happens automatically when importing from
        ``lalamo.speculator``.
        """
        subcls = cls._registry.get(name)
        if subcls is None:
            known = ", ".join(cls._registry)
            raise ValueError(f"Unknown drafter {name!r}. Registered: {known}")
        return subcls.deserialize_impl(data, **kwargs)

    @classmethod
    @abstractmethod
    def deserialize_impl(cls, data: bytes, **kwargs: object) -> Self: ...

    @abstractmethod
    def draft(self, lm: "LMState", seed: int) -> TrieNode:
        """Root token must be ``lm.bonus``. Children are continuations after it."""
        ...

    def on_prefill(self, lm: "LMState") -> "Drafter":  # noqa: ARG002
        """Called once after prefill, before the first draft()."""
        return self

    def update_after_verify(
        self,
        prev_lm: "LMState",  # noqa: ARG002
        accepted: list[int],  # noqa: ARG002
        bonus: int,  # noqa: ARG002
        new_lm: "LMState",  # noqa: ARG002
    ) -> "Drafter":
        """Post-verify lifecycle hook. Returns updated drafter."""
        return self

    @abstractmethod
    def serialize(self) -> bytes: ...

    @staticmethod  # noqa: B027
    def train_command(app: Typer, callbacks_type: type) -> None:
        """Override to register a training CLI command on app."""
