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
    """Immutable snapshot after the last verified token.

    ``hiddens`` stores per-layer hidden states at the head position.
    The last entry (``hiddens[-1]``) is the output-norm result — equivalent
    to the old ``h_i``.  Earlier entries are intermediate layer outputs,
    available for EAGLE-style drafters that condition on multiple layers.
    """

    kv_cache: State
    hiddens: tuple[jnp.ndarray, ...]  # per-layer (d,); hiddens[-1] = output norm
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
