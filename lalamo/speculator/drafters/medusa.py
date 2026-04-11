import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.speculator.drafter import Drafter
from lalamo.speculator.speculate import GumbelSeed, LMState
from lalamo.speculator.trie import TrieNode
from lalamo.speculator.utils import top_k_from_logits


class MedusaHeads(eqx.Module):
    heads: tuple[eqx.nn.Linear, ...]
    num_heads: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    def __init__(self, num_heads: int, d_model: int, vocab_size: int, *, key: jax.random.PRNGKey) -> None:
        keys = jax.random.split(key, num_heads)
        self.heads = tuple(eqx.nn.Linear(d_model, vocab_size, use_bias=False, key=k) for k in keys)
        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size = vocab_size

    def predict(self, hidden: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        return tuple(head(hidden) for head in self.heads)


@Drafter.register("medusa")
@dataclass(frozen=True)
class MedusaDrafter(Drafter):
    """Builds a Cartesian-product tree from K independent linear heads."""

    heads: MedusaHeads
    width: int = 4

    @property
    def depth(self) -> int:
        return self.heads.num_heads

    def draft(self, lm: LMState, seed: int) -> TrieNode:
        hidden = lm.hiddens[-1]
        all_logits = self.heads.predict(hidden)
        candidates_per_head = [top_k_from_logits(logits, self.width) for logits in all_logits]

        gseed = GumbelSeed(seed)
        root = TrieNode(token=lm.bonus, seed=gseed.derive(1).value)
        self._expand(root, candidates_per_head, head_idx=0, gseed=gseed)
        return root

    def _expand(
        self,
        node: TrieNode,
        candidates_per_head: list[list[int]],
        head_idx: int,
        gseed: GumbelSeed,
    ) -> None:
        if head_idx >= len(candidates_per_head):
            return
        node_seed = gseed.derive(head_idx + 2).value
        for tok in candidates_per_head[head_idx]:
            child = node.add_child(tok, seed=node_seed)
            self._expand(child, candidates_per_head, head_idx + 1, gseed)

    def serialize(self) -> bytes:
        header = struct.pack("<3I", self.heads.num_heads, self.heads.d_model, self.heads.vocab_size)
        params = [p for p in jax.tree.leaves(self.heads) if isinstance(p, jnp.ndarray)]
        body = b"".join(np.asarray(p).astype(np.float32).tobytes() for p in params)
        return header + body

    @classmethod
    def deserialize_impl(cls, data: bytes, **kwargs: object) -> Self:
        width = int(kwargs.get("width", 4))
        num_heads, d_model, vocab_size = struct.unpack("<3I", data[:12])
        heads = MedusaHeads(num_heads, d_model, vocab_size, key=jax.random.key(0))
        offset = 12
        leaves, treedef = jax.tree.flatten(heads)
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, jnp.ndarray):
                size = leaf.size * 4
                buf = np.frombuffer(data[offset : offset + size], dtype=np.float32)
                new_leaves.append(jnp.array(buf.reshape(leaf.shape)))
                offset += size
            else:
                new_leaves.append(leaf)
        heads = treedef.unflatten(new_leaves)
        return cls(heads=heads, width=width)


class MedusaTrainingEvent:
    __slots__ = ("loss", "step")

    def __init__(self, step: int, loss: float) -> None:
        self.step = step
        self.loss = loss


def train_medusa(
    traces: Iterable[LalamoCompletion],
    d_model: int,
    vocab_size: int,
    num_heads: int = 4,
    width: int = 4,
    learning_rate: float = 1e-3,
    numepochs: int = 1,
    progress_callback: Callable[[MedusaTrainingEvent], None] | None = None,
    key: jax.random.PRNGKey | None = None,
) -> Self:
    if key is None:
        key = jax.random.key(0)

    heads = MedusaHeads(num_heads, d_model, vocab_size, key=key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(heads, eqx.is_array))

    hiddens_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for trace in traces:
        h = np.asarray(trace.activation_output)
        toks = np.asarray(trace.completion_token_ids)
        seq_len = len(toks)
        if seq_len <= num_heads:
            continue
        valid_len = seq_len - num_heads
        hiddens_list.append(h[:valid_len])
        targets = np.stack([toks[k + 1 : k + 1 + valid_len] for k in range(num_heads)], axis=1)
        targets_list.append(targets)

    if not hiddens_list:
        return MedusaDrafter(heads=heads, width=width)

    all_hiddens = jnp.array(np.concatenate(hiddens_list, axis=0))
    all_targets = jnp.array(np.concatenate(targets_list, axis=0))

    @eqx.filter_jit
    def loss_fn(heads: MedusaHeads, hiddens: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        def per_sample(h: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
            logits = heads.predict(h)
            losses = jnp.stack(
                [optax.softmax_cross_entropy_with_integer_labels(logits[k], t[k]) for k in range(heads.num_heads)]
            )
            return jnp.mean(losses)

        return jnp.mean(jax.vmap(per_sample)(hiddens, targets))

    @eqx.filter_jit
    def step(
        heads: MedusaHeads,
        opt_state: optax.OptState,
        hiddens: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[MedusaHeads, optax.OptState, jnp.ndarray]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(heads, hiddens, targets)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(heads, eqx.is_array))
        heads = eqx.apply_updates(heads, updates)
        return heads, opt_state, loss

    batch_size = min(1024, len(all_hiddens))
    num_batches = (len(all_hiddens) + batch_size - 1) // batch_size
    global_step = 0

    for _epoch in range(numepochs):
        perm = jax.random.permutation(jax.random.key(global_step), len(all_hiddens))
        shuffled_h = all_hiddens[perm]
        shuffled_t = all_targets[perm]
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_hiddens))
            heads, opt_state, loss = step(heads, opt_state, shuffled_h[start:end], shuffled_t[start:end])
            global_step += 1
            if progress_callback is not None:
                progress_callback(MedusaTrainingEvent(global_step, float(loss)))

    return MedusaDrafter(heads=heads, width=width)
