"""Drafter protocol and core state for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import jax.numpy as jnp

    from lalamo.modules.token_mixers.state.common import State
    from lalamo.speculator.trie import TrieNode


@dataclass(frozen=True)
class LMState:
    """Immutable snapshot after the last verified token."""

    kv_cache: State
    h_i: jnp.ndarray  # (d,)     — hidden state at head
    logits: jnp.ndarray  # (vocab,) — next-token distribution at head
    position: int  # tokens written to the KV cache so far
    context: tuple[int, ...] = ()  # verified token history (n-gram suffix lookup)


@dataclass(frozen=True)
class SamplerConfig:
    width: int = 4  # max children per trie node
    K: int = 8  # max speculation depth


class Drafter(Protocol):
    """Produce a draft tree from the current LM state.

    ``last_token`` is the bonus token from the previous verify step
    (= ``lm.context[-1]``). Passed explicitly so neural drafters can
    index into the embedding table without slicing context.
    """

    def draft(self, lm: LMState, last_token: int, seed: int) -> TrieNode: ...
