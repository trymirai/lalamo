import inspect
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Self

from jaxtyping import Array, Float
from typer import Typer

from lalamo.modules.decoder import Decoder
from lalamo.modules.token_mixers.state.common import State
from lalamo.registry_abc import RegistryABC


@dataclass(frozen=True)
class LMState:
    """Immutable snapshot after the last verified position.

    Invariants:
    - ``logits`` is the next-token distribution at the head position.
    - ``bonus`` is sampled from ``logits`` and is always present.
    - The next verify batch starts with ``bonus`` as its first token.
    - ``bonus`` is NOT yet materialized in the KV cache.
    - ``layer_outputs`` has one ``(suffix, d)`` array per layer in the
      speculator's ``trace_layer_outputs``. After prefill ``suffix`` equals
      ``prompt_len`` (or ``prefill_hidden_range`` if specified);
      after verify ``suffix == 1 + num_accepted`` (bonus + accepted rows).
    - ``output_norm`` is populated iff ``trace_output_norm`` is True;
      same ``suffix`` semantics as ``layer_outputs``.
    """

    kv_cache: State
    layer_outputs: tuple[Float[Array, "suffix channels"], ...]
    output_norm: Float[Array, "suffix channels"] | None
    logits: Float[Array, " vocab"]
    position: int  # tokens written to the KV cache so far
    bonus: int  # sampled next token, always present


@dataclass(frozen=True)
class SamplerConfig:
    width: int = 4  # max children per trie node
    K: int = 8  # max speculation depth
    max_tokens: int = 2048  # max generation length
    seed: int = 42  # initial RNG seed (used to construct Speculator.seed)


@dataclass(frozen=True, kw_only=True)
class SpeculationStep:
    """Per-step summary yielded by :meth:`Speculator.step`."""

    accepted: list[int]
    bonus: int


@dataclass(frozen=True, kw_only=True)
class Speculator[P](RegistryABC):
    """Abstract base. Lifecycle is :meth:`prefill` then repeated :meth:`step`.
    Subclasses set ``name: ClassVar[str]`` for :class:`RegistryABC` lookup.
    """

    decoder: Decoder
    config: SamplerConfig
    eos_set: frozenset[int]
    trace_layer_outputs: tuple[int, ...] | None = None
    trace_output_norm: bool = False
    prefill_hidden_range: int | None = None

    name: ClassVar[str]

    @classmethod
    def deserialize(
        cls,
        name: str,
        data: bytes,
        *,
        decoder: Decoder,
        config: SamplerConfig,
        eos_set: frozenset[int],
        **extra: object,
    ) -> "Speculator":
        """Deserialize a speculator by name from a binary blob.

        Requires that the speculator module has been imported so the subclass
        is registered (happens automatically when importing from ``lalamo.speculator``).
        """
        for subcls in cls.registered_types():
            if subcls.name == name:
                return subcls.deserialize_impl(
                    data,
                    decoder=decoder,
                    config=config,
                    eos_set=eos_set,
                    **extra,
                )
        known = ", ".join(sorted(s.name for s in cls.registered_types()))
        raise ValueError(f"Unknown speculator {name!r}. Registered: {known}")

    @classmethod
    def registered_types(cls) -> Iterator[type["Speculator"]]:
        """Iterate over concrete (non-abstract) registered speculator subclasses."""
        return (subcls for subcls in cls.__descendants__() if not inspect.isabstract(subcls))

    @classmethod
    def deserialize_impl(
        cls,
        data: bytes,
        *,
        decoder: Decoder,
        config: SamplerConfig,
        eos_set: frozenset[int],
        **extra: object,
    ) -> Self:
        raise NotImplementedError(f"{cls.__name__} does not support deserialization")

    def serialize(self) -> bytes:
        raise NotImplementedError(f"{type(self).__name__} does not support serialization")

    @abstractmethod
    def draft(self, lm: LMState) -> P:
        """Build a proposal consumed by :meth:`step`. ``P`` is drafter-specific:
        ``TreeSpeculator`` fixes ``P = TrieNode``; a chain-based speculator would
        fix ``P`` to its own chain struct.
        """

    @abstractmethod
    def step(self, lm: LMState) -> tuple[Self, LMState, SpeculationStep]:
        """Advance one verify iteration.

        Returns ``(new_self, new_lm, step_summary)`` where:
        - ``new_self`` has drafter-specific state (RNG, NGram context, etc.)
          updated for the next step;
        - ``new_lm`` is the state after accepting drafts + sampling next bonus;
        - ``step_summary`` is the public per-step record yielded to the caller.
        """

    @abstractmethod
    def prefill(self, prompt_ids: list[int]) -> tuple[Self, LMState]:
        """Initialize ``LMState`` from the prompt. Returns ``(new_self, lm_state)``.
        Subclasses own the whole forward pass + bonus sampling.
        """

    @property
    def generation_capacity(self) -> int:
        return self.config.max_tokens + self.config.K * self.config.width + 16

    @staticmethod
    def train_command(app: Typer, callbacks_type: type) -> None:
        """Register a training subcommand on ``app`` if this speculator has trainable state.

        Default is a no-op. Speculators without any trainable parameters (pure
        heuristics, rule-based lookup, etc.) need not override.
        """
