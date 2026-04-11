"""NGramDrafter: adapts NGramSpeculator to the Drafter protocol.

Builds a draft trie using n-gram probability lookups for candidate
selection and LM logits for the first level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lalamo.speculator.speculate import _derive_seed
from lalamo.speculator.trie import TrieNode

if TYPE_CHECKING:
    import jax.numpy as jnp

    from lalamo.speculator.drafter import LMState
    from lalamo.speculator.ngram import NGramSpeculator


def _top_k_from_logits(logits: jnp.ndarray, k: int) -> list[int]:
    """Return top-k token IDs from logits (host-side)."""
    logits_np = np.asarray(logits)
    if k >= logits_np.shape[0]:
        return list(np.argsort(logits_np)[::-1])
    indices = np.argpartition(logits_np, -k)[-k:]
    return list(indices[np.argsort(logits_np[indices])[::-1]])


def _top_k_from_probs(probs: dict[int, float], k: int) -> list[int]:
    """Return top-k token IDs from probability dict."""
    sorted_items = sorted(probs.items(), key=lambda x: -x[1])
    return [tok for tok, _ in sorted_items[:k]]


@dataclass(frozen=True)
class NGramDrafter:
    """Drafter that uses NGramSpeculator for tree construction.

    Level 0: candidates come from LM logits (top-width).
    Level 1+: candidates come from speculator.probs(context).
    Greedy chain follows the top-1 candidate at each level,
    with width fan-out at every depth.
    """

    speculator: NGramSpeculator
    width: int = 4
    depth: int = 8

    def draft(self, lm: LMState, last_token: int, seed: int) -> TrieNode:
        root = TrieNode(token=last_token, seed=_derive_seed(seed, 1))

        # Level 0: use LM logits for first-level candidates
        candidates = _top_k_from_logits(lm.logits, self.width)
        node_seed = _derive_seed(seed, 2)
        for tok in candidates:
            root.add_child(tok, seed=node_seed)

        # Greedy chain: follow top-1 candidate, fan out at each depth
        chain_node = root.get_child(candidates[0])
        context = list(lm.context) + [candidates[0]]

        for k in range(1, self.depth):
            probs = self.speculator.probs(context)
            if not probs:
                break

            next_candidates = _top_k_from_probs(probs, self.width)
            node_seed = _derive_seed(seed, k + 2)
            for tok in next_candidates:
                chain_node.add_child(tok, seed=node_seed)

            chain_node = chain_node.get_child(next_candidates[0])
            context.append(next_candidates[0])

        return root
