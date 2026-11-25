"""The draft tree of the tree speculators: expansion of a frontier by rounds, then pruning to a verification budget.

A step drafts a tree below a root, the last committed token. The drafter gives every depth d = 1 .. rows a pool of
candidate tokens with logits `a`, and every depth has one noise row `G` over its pool: the noise the target's sampler
will add at that position, shared by all nodes of the depth. A node-dependent correction `rho` (a function of the
node and its ancestors, supplied by the caller) turns the pool logits of the node's depth into the logits of its
children, l = a + rho.

  rounds   Round 0 expands the root; every later round expands the `expand` frontier entries with the largest key.
           The key is the path weight -- the sum of min(l_c - logsumexp(l), 0) along the path -- as a monotone
           unsigned integer without its lowest bit, with "may still be expanded" (depth < rows) in the top bit;
           ties go to the smaller parent slot, then to the smaller token, then to the smaller entry.
  children An expanded node (a slot) gets the `children` best pool candidates by l + G, ties to the smaller pool
           index, as the frontier entries slot * children .. slot * children + children - 1.
  pruning  The nodes are the slots followed by the entries that never became slots. The `budget` of them with the
           largest pruning score are kept for verification; the score is the same path sum taken on l + G / sigma,
           and ties go to the smaller index. The root has the largest possible score and is always kept; a kept
           node's parent is kept.

The construction does not depend on the drafter: it reads pool logits, noise, pool tokens and `correction_of`.

A round always makes `expand` slots. When the frontier holds fewer live entries than that (a shape with expand >
children, or so few rows that the leaves run out), the round takes entries that are not live: unwritten ones give
slots with a pruning score of minus infinity, which are never kept, and entries that are slots already are expanded
a second time, so the same node appears twice. The default shape never runs out of live entries.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, UInt

__all__ = [
    "CorrectionOf",
    "FrontierTree",
    "FrontierTreeShape",
    "KeptNodes",
    "accepted_length",
    "build_frontier_tree",
    "kept_nodes",
]

# (carry, new slots [batch, n], their tokens [batch, n], their depths [batch, n], their ancestor slots by depth
# [batch, n, rows] with -1 past the parent) -> (correction of the pools of the new slots [batch, n, pool], carry)
type CorrectionOf[Carry] = Callable[
    [Carry, Int[Array, "batch n"], Int[Array, "batch n"], Int[Array, "batch n"], Int[Array, "batch n rows"]],
    tuple[Float[Array, "batch n pool"], Carry],
]


@dataclass(frozen=True)
class FrontierTreeShape:
    slots: int = 61  # expanded nodes, the root included
    children: int = 4  # frontier entries an expanded node gets
    expand: int = 4  # nodes expanded per round after the root's
    budget: int = 16  # nodes kept for verification, the root included
    prune_sigma: float = 1.5

    def __post_init__(self) -> None:
        if self.slots < 1 or self.children < 1 or self.expand < 1 or self.budget < 1:
            raise ValueError("slots, children, expand and budget must be positive")
        if (self.slots - 1) % self.expand != 0:
            raise ValueError(
                f"slots - 1 must be a multiple of expand, got {self.slots} slots and expand {self.expand}"
            )
        if self.budget > self.slots * (1 + self.children):
            raise ValueError("the budget exceeds the number of nodes a tree can hold")
        if self.prune_sigma <= 0:
            raise ValueError(f"prune_sigma must be positive, got {self.prune_sigma}")

    @property
    def entries(self) -> int:
        return self.slots * self.children

    @property
    def rounds(self) -> int:
        return 1 + (self.slots - 1) // self.expand


class FrontierTree(NamedTuple):
    """A built tree: per slot, and per frontier entry (the children of slot s are entries s * children ..)."""

    slot_token: Int[Array, "batch slots"]
    slot_depth: Int[Array, "batch slots"]
    slot_entry: Int[Array, "batch slots"]  # the entry the slot was made of, -1 for the root
    slot_prune: Float[Array, "batch slots"]
    slot_correction: Float[Array, "batch slots pool"]
    entry_pool: Int[Array, "batch entries"]  # index of the entry's token in the pool of its depth
    entry_token: Int[Array, "batch entries"]
    entry_active: Bool[Array, "batch entries"]  # a node that was never expanded
    entry_prune: Float[Array, "batch entries"]


class KeptNodes(NamedTuple):
    """The nodes kept for verification, parents before children; node 0 is the root."""

    token: Int[Array, "batch budget"]
    depth: Int[Array, "batch budget"]
    parent: Int[Array, "batch budget"]  # index among the kept nodes, -1 for the root and for invalid nodes
    valid: Bool[Array, "batch budget"]  # False where the tree holds fewer nodes than the budget


def _score_key(scores: Float[Array, "..."]) -> UInt[Array, "..."]:
    """A monotone unsigned integer of a float32: a larger float gives a larger key."""
    bits = jax.lax.bitcast_convert_type(scores, jnp.uint32)
    return jnp.where((bits & jnp.uint32(0x80000000)) == 0, bits ^ jnp.uint32(0x80000000), ~bits)


def build_frontier_tree[Carry](
    shape: FrontierTreeShape,
    pool_logits: Float[Array, "batch rows pool"],
    pool_noise: Float[Array, "batch rows pool"],
    pool_tokens: Int[Array, "batch rows pool"],
    root_token: Int[Array, " batch"],
    correction_of: CorrectionOf[Carry],
    carry: Carry,
) -> FrontierTree:
    """The tree of the module docstring. Row d - 1 of the pools belongs to depth d, the children of depth d - 1.

    `correction_of` is called once per round with the slots the round creates (the root alone in round 0).
    """
    batch, rows, pool_size = pool_logits.shape
    slots, children, entries = shape.slots, shape.children, shape.entries
    index = jnp.arange(batch)[:, None]
    slot_token = jnp.zeros((batch, slots), jnp.int32).at[:, 0].set(root_token)
    slot_depth = jnp.zeros((batch, slots), jnp.int32)
    slot_entry = jnp.full((batch, slots), -1, jnp.int32)
    slot_weight = jnp.zeros((batch, slots), jnp.float32)
    slot_prune = jnp.zeros((batch, slots), jnp.float32)
    slot_correction = jnp.zeros((batch, slots, pool_size), jnp.float32)
    ancestors = jnp.full((batch, slots, rows), -1, jnp.int32)
    entry_pool = jnp.zeros((batch, entries), jnp.int32)
    entry_token = jnp.zeros((batch, entries), jnp.int32)
    entry_depth = jnp.zeros((batch, entries), jnp.int32)
    entry_weight = jnp.full((batch, entries), -jnp.inf, jnp.float32)
    entry_prune = jnp.full((batch, entries), -jnp.inf, jnp.float32)
    entry_active = jnp.zeros((batch, entries), bool)
    new = jnp.zeros((batch, 1), jnp.int32)
    for round_index in range(shape.rounds):
        if round_index:
            key = jnp.where(
                entry_active & jnp.isfinite(entry_weight),
                ((entry_depth < rows).astype(jnp.uint32) << 31) | (_score_key(entry_weight) >> 1),
                jnp.uint32(0),
            )
            entry_parent = jnp.arange(entries, dtype=jnp.int32) // children
            # lexsort sorts by its last key first: key descending, then parent, token and entry ascending.
            order = jnp.lexsort(
                (
                    jnp.broadcast_to(jnp.arange(entries), key.shape),
                    entry_token,
                    jnp.broadcast_to(entry_parent, key.shape),
                    ~key,
                ),
                axis=-1,
            )
            picked = order[:, : shape.expand]
            first = 1 + (round_index - 1) * shape.expand
            new = jnp.broadcast_to(first + jnp.arange(shape.expand, dtype=jnp.int32)[None, :], (batch, shape.expand))
            parent_slot = picked // children
            slot_token = slot_token.at[index, new].set(jnp.take_along_axis(entry_token, picked, axis=1))
            slot_depth = slot_depth.at[index, new].set(jnp.take_along_axis(entry_depth, picked, axis=1))
            slot_entry = slot_entry.at[index, new].set(picked)
            slot_weight = slot_weight.at[index, new].set(jnp.take_along_axis(entry_weight, picked, axis=1))
            slot_prune = slot_prune.at[index, new].set(jnp.take_along_axis(entry_prune, picked, axis=1))
            parent_ancestors = jnp.take_along_axis(ancestors, parent_slot[..., None], axis=1)
            parent_depth = jnp.take_along_axis(slot_depth, parent_slot, axis=1)
            own_ancestors = parent_ancestors.at[
                index, jnp.arange(shape.expand)[None, :], jnp.minimum(parent_depth, rows - 1)
            ].set(parent_slot)
            ancestors = ancestors.at[index, new].set(own_ancestors)
            entry_active = entry_active.at[index, picked].set(False)
        depth = jnp.take_along_axis(slot_depth, new, axis=1)
        tokens = jnp.take_along_axis(slot_token, new, axis=1)
        row = jnp.minimum(depth, rows - 1)
        correction, carry = correction_of(
            carry, new, tokens, depth, jnp.take_along_axis(ancestors, new[..., None], axis=1)
        )
        slot_correction = slot_correction.at[index, new].set(correction)
        logits = jnp.take_along_axis(pool_logits, row[..., None], axis=1) + correction
        noise = jnp.take_along_axis(pool_noise, row[..., None], axis=1)
        prune_logits = logits + noise / shape.prune_sigma
        top_value, top_index = jax.lax.top_k(logits + noise, children)
        edge = jnp.minimum(
            jnp.take_along_axis(logits, top_index, axis=-1) - jax.nn.logsumexp(logits, axis=-1, keepdims=True), 0.0
        )
        prune_edge = jnp.minimum(
            jnp.take_along_axis(prune_logits, top_index, axis=-1)
            - jax.nn.logsumexp(prune_logits, axis=-1, keepdims=True),
            0.0,
        )
        parent_weight = jnp.take_along_axis(slot_weight, new, axis=1)[..., None]
        parent_prune = jnp.take_along_axis(slot_prune, new, axis=1)[..., None]
        valid = (depth < rows)[..., None] & jnp.isfinite(top_value)
        targets = rearrange(new[..., None] * children + jnp.arange(children), "batch n children -> batch (n children)")

        def flat(values: Array) -> Array:
            return rearrange(values, "batch n children -> batch (n children)")

        child_tokens = jnp.take_along_axis(
            jnp.take_along_axis(pool_tokens, row[..., None], axis=1), top_index, axis=-1
        )
        entry_pool = entry_pool.at[index, targets].set(flat(top_index))
        entry_token = entry_token.at[index, targets].set(flat(child_tokens))
        entry_depth = entry_depth.at[index, targets].set(flat(jnp.broadcast_to(depth[..., None] + 1, top_index.shape)))
        entry_weight = entry_weight.at[index, targets].set(
            flat(jnp.where(valid, jnp.minimum(parent_weight, parent_weight + edge), -jnp.inf))
        )
        entry_prune = entry_prune.at[index, targets].set(
            flat(jnp.where(valid, jnp.minimum(parent_prune, parent_prune + prune_edge), -jnp.inf))
        )
        entry_active = entry_active.at[index, targets].set(flat(valid))
    return FrontierTree(
        slot_token,
        slot_depth,
        slot_entry,
        slot_prune,
        slot_correction,
        entry_pool,
        entry_token,
        entry_active,
        entry_prune,
    )


def kept_nodes(shape: FrontierTreeShape, tree: FrontierTree) -> KeptNodes:
    """The `budget` nodes with the largest pruning score, ties to the smaller node index (slots, then entries).

    Returned in the order of that index, which puts every parent before its children and the root first.
    """
    batch, slots = tree.slot_token.shape
    children, entries = shape.children, shape.entries
    index = jnp.arange(batch)[:, None]
    scores = jnp.concatenate([tree.slot_prune, jnp.where(tree.entry_active, tree.entry_prune, -jnp.inf)], axis=1)
    best = jnp.argsort(-scores, axis=1, stable=True)[:, : shape.budget]
    kept = jnp.sort(best, axis=1)
    valid = jnp.isfinite(jnp.take_along_axis(scores, kept, axis=1))
    entry_parent = jnp.broadcast_to(jnp.arange(entries, dtype=jnp.int32) // children, (batch, entries))
    slot_parent = jnp.where(tree.slot_entry >= 0, tree.slot_entry // children, -1)
    parent_slot = jnp.concatenate([slot_parent, entry_parent], axis=1)
    entry_depth = jnp.take_along_axis(tree.slot_depth, entry_parent, axis=1) + 1
    position = (
        jnp.full((batch, slots + entries), -1, jnp.int32)
        .at[index, kept]
        .set(jnp.broadcast_to(jnp.arange(shape.budget, dtype=jnp.int32), kept.shape))
    )
    kept_parent_slot = jnp.take_along_axis(parent_slot, kept, axis=1)
    parent = jnp.where(
        valid & (kept_parent_slot >= 0),
        jnp.take_along_axis(position, jnp.maximum(kept_parent_slot, 0), axis=1),
        -1,
    )
    return KeptNodes(
        token=jnp.take_along_axis(jnp.concatenate([tree.slot_token, tree.entry_token], axis=1), kept, axis=1),
        depth=jnp.take_along_axis(jnp.concatenate([tree.slot_depth, entry_depth], axis=1), kept, axis=1),
        parent=parent,
        valid=valid,
    )


def accepted_length(nodes: KeptNodes, sampled: Int[Array, "batch budget"]) -> Int[Array, " batch"]:
    """How many drafted tokens the target accepts: the length of the path from the root along which every node's
    child carries the token the target sampled after that node (`sampled[:, i]` for kept node i).

    The root is not counted; the token the target samples after the last accepted node is extra, as usual.
    """
    batch, budget = nodes.token.shape
    current = jnp.zeros((batch,), jnp.int32)
    length = jnp.zeros((batch,), jnp.int32)
    alive = nodes.valid[:, 0]
    for _ in range(budget - 1):
        wanted = jnp.take_along_axis(sampled, current[:, None], axis=1)
        follows = nodes.valid & (nodes.parent == current[:, None]) & (nodes.token == wanted)
        found = follows.any(axis=1) & alive
        current = jnp.where(found, jnp.argmax(follows, axis=1).astype(jnp.int32), current)
        length = length + found.astype(jnp.int32)
        alive = found
    return length
