"""Speculative decoding pipeline: draft tree → verify → advance.

Key design:
  - LMState is frozen: every step produces a NEW snapshot.
  - SpeculationContext is frozen: pure methods, no mutation.
  - SpeculationRun is the only mutable object (holds context list, seed, result).
  - Draft mechanism is abstracted via the Drafter protocol.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.token_mixers.state.common import State
from lalamo.modules.token_mixers.state.kv_cache import StaticKVCacheLayer
from lalamo.speculator.drafter import Drafter, LMState, SamplerConfig
from lalamo.speculator.trie import FlatTrie, TrieNode

if TYPE_CHECKING:
    from lalamo.modules.decoder import Decoder, DecoderActivationTrace


def _extract_hiddens(
    trace: DecoderActivationTrace, batch: int, token: int,
) -> tuple[jnp.ndarray, ...]:
    """Extract per-layer hidden states + output norm at a single position."""
    return tuple(
        lr.outputs[batch, token] for lr in trace.layer_results
    ) + (trace.output_norm[batch, token],)


# ── Data types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class SpeculationStep:
    accepted: list[int]
    bonus: int
    proposed: int
    depth: int


@dataclass
class SpeculativeDecodingResult:
    num_steps: int = 0
    total_accepted: int = 0
    total_proposed: int = 0
    generated: list[int] = field(default_factory=list)

    @property
    def mean_accepted_length(self) -> float:
        return self.total_accepted / max(self.num_steps, 1)

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / max(self.total_proposed, 1)

    @property
    def tokens_per_step(self) -> float:
        return len(self.generated) / max(self.num_steps, 1)


# ── Pure helpers ────────────────────────────────────────────────


def _derive_seed(seed: int, depth: int) -> int:
    """splitmix64-style hash: full avalanche even for small depth values."""
    h = seed + depth * 2654435761
    h = ((h >> 16) ^ h) * 0x45D9F3B37197344D & 0xFFFFFFFFFFFFFFFF
    h = ((h >> 16) ^ h) * 0x45D9F3B37197344D & 0xFFFFFFFFFFFFFFFF
    return ((h >> 16) ^ h) & 0xFFFFFFFFFFFFFFFF


def _next_seed(seed: int, context_len: int) -> int:
    return (seed * 2654435761 ^ context_len) & 0xFFFFFFFFFFFFFFFF


def _jax_gumbel_sample_batch(
    logits: jnp.ndarray,
    seeds: jnp.ndarray,
) -> jnp.ndarray:
    """Batched Gumbel-max sampling on device. (N, V) logits, (N,) seeds → (N,) tokens."""
    keys = jax.vmap(lambda s: jax.random.key(s))(seeds.astype(jnp.uint32))
    noise = jax.vmap(lambda k: jax.random.gumbel(k, (logits.shape[1],), dtype=jnp.float32))(keys)
    return jnp.argmax(logits + noise, axis=-1)


@functools.partial(jax.jit, static_argnames=("max_slots",))
def _compact_kv_cache(
    state: State,
    cache_len: jnp.ndarray,
    accepted_indices: jnp.ndarray,
    num_accepted: jnp.ndarray,
    max_slots: int,
) -> State:
    """Keep accepted children's KV entries, discard rejected draft tokens.

    JIT-compiled: all layers x 2 scatter ops fuse into a single XLA program.
    ``cache_len`` and ``num_accepted`` are JAX scalars (not Python ints)
    to avoid recompilation when values change between steps.
    """
    dst = jnp.arange(max_slots, dtype=jnp.int32) + cache_len
    src = cache_len + accepted_indices
    valid = jnp.arange(max_slots) < num_accepted
    src = jnp.where(valid, src, dst)
    new_length = cache_len + num_accepted

    def compact_layer(layer: StaticKVCacheLayer) -> StaticKVCacheLayer:
        return StaticKVCacheLayer(
            has_sinks=layer.has_sinks,
            keys=layer.keys.at[:, dst].set(layer.keys[:, src]),
            values=layer.values.at[:, dst].set(layer.values[:, src]),
            current_length=jnp.full_like(layer.current_length, new_length),
        )

    return State(compact_layer(layer) for layer in state)


# ── Speculation context (frozen) ────────────────────────────────


@dataclass(frozen=True)
class SpeculationContext:
    """Immutable environment: model weights, drafter, and configuration.

    All methods are pure — they take ``LMState`` in and return a new one.
    """

    decoder: Decoder
    drafter: Drafter
    lm_w: jnp.ndarray  # (V, d) unembedding weights
    embed_w: jnp.ndarray  # (V, d) embedding weights
    config: SamplerConfig
    eos_set: frozenset[int]
    use_gumbel: bool = True

    @classmethod
    def create(
        cls,
        decoder: Decoder,
        drafter: Drafter,
        config: SamplerConfig,
        eos_set: set[int],
        use_gumbel: bool = True,
    ) -> SpeculationContext:
        lm_w = jnp.array(np.asarray(decoder.embedding.unembedding_matrix).astype(np.float32))
        embed_w = jnp.array(np.asarray(decoder.embedding.embedding_matrix).astype(np.float32))
        return cls(
            decoder=decoder,
            drafter=drafter,
            lm_w=lm_w,
            embed_w=embed_w,
            config=config,
            eos_set=frozenset(eos_set),
            use_gumbel=use_gumbel,
        )

    # ── Prefill ─────────────────────────────────────────────

    def prefill(self, prompt_ids: list[int], capacity: int) -> LMState:
        state = self.decoder.init_static_state(1, capacity)
        prefix = jnp.array([prompt_ids], dtype=jnp.int32)
        fwd = self.decoder(
            prefix,
            jnp.arange(len(prompt_ids))[None, :],
            state,
            return_updated_state=True,
            return_activation_trace=True,
        )
        return LMState(
            kv_cache=fwd.updated_state,
            hiddens=_extract_hiddens(fwd.activation_trace, 0, -1),
            logits=fwd.logits[0, -1],
            position=len(prompt_ids),
            context=tuple(prompt_ids),
        )

    # ── Single-token advance ──────────────────────────────

    def advance_one(self, lm: LMState, token: int) -> LMState:
        """Forward a single token through the decoder, returning fresh state."""
        pos = lm.position
        fwd = self.decoder(
            jnp.array([[token]], dtype=jnp.int32),
            jnp.array([[pos]], dtype=jnp.int32),
            lm.kv_cache,
            return_updated_state=True,
            return_activation_trace=True,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        )
        return LMState(
            kv_cache=fwd.updated_state,
            hiddens=_extract_hiddens(fwd.activation_trace, 0, 0),
            logits=fwd.logits[0, 0],
            position=pos + 1,
            context=lm.context + (token,),
        )

    # ── Verify ──────────────────────────────────────────────

    def verify(
        self,
        lm: LMState,
        trie: TrieNode,
    ) -> tuple[list[int], tuple[int, int], LMState]:
        """Tree forward (children only) → device-side sample → accept → compact.

        All device arrays are padded to fixed shapes (``width * K``) to
        prevent XLA recompilation when tree size varies between steps.

        Returns ``(accepted_tokens, (bonus_token, bonus_seed), new_lm_state)``.
        """
        flat = trie.linearize(include_root=False)
        if flat.num_nodes == 0:
            bonus_token = int(jnp.argmax(lm.logits))
            return [], (bonus_token, trie.seed), lm

        cache_len = int(lm.kv_cache[0].current_length[0])
        num_nodes = flat.num_nodes
        max_nodes = self.config.width * self.config.K

        def _pad(arr: np.ndarray, n: int, fill: int = 0) -> np.ndarray:
            pad_len = n - len(arr)
            if pad_len <= 0:
                return arr[:n]
            return np.concatenate([arr, np.full(pad_len, fill, dtype=arr.dtype)])

        tok_ids = jnp.array(
            _pad(flat.token_ids, max_nodes, 0)[None, :],
            dtype=jnp.int32,
        )
        positions = jnp.array(
            _pad(flat.positions(cache_len), max_nodes, cache_len)[None, :],
            dtype=jnp.int32,
        )
        parent_indices = jnp.array(
            _pad(flat.parent_indices, max_nodes, -1)[None, :],
            dtype=jnp.int32,
        )

        fwd = self.decoder(
            tok_ids,
            positions,
            lm.kv_cache,
            return_updated_state=True,
            return_activation_trace=True,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            attention_parent_indices=parent_indices,
            attention_max_depth=self.config.K,
        )

        # ── Sampling (on device, fixed shape) ────────────────
        all_logits = jnp.concatenate(
            [lm.logits[None, :], fwd.logits[0]],
            axis=0,
        ).astype(jnp.float32)

        all_seeds_real = np.concatenate(
            [np.array([trie.seed], dtype=np.uint64), flat.seeds],
        )
        all_seeds = _pad(all_seeds_real, max_nodes + 1, 0)

        if self.use_gumbel:
            sampled_all = np.asarray(
                _jax_gumbel_sample_batch(
                    all_logits,
                    jnp.array(all_seeds & 0xFFFFFFFF, dtype=jnp.uint32),
                )
            )
        else:
            sampled_all = np.asarray(jnp.argmax(all_logits, axis=-1))

        sampled_tokens = sampled_all[: num_nodes + 1]

        # ── Host-side acceptance walk ───────────────────────
        full_flat = FlatTrie(
            token_ids=np.concatenate([[trie.token], flat.token_ids]),
            seeds=all_seeds_real,
            depths=np.concatenate([[0], flat.depths + 1]),
            parent_indices=np.concatenate([[-1], flat.parent_indices + 1]),
        )
        accepted_tokens, accepted_indices = full_flat.accept(sampled_tokens)
        num_accepted = len(accepted_tokens)

        # ── KV compact (fixed shape, JIT-compiled) ────────
        max_accepted = self.config.K
        fwd_indices = accepted_indices - 1
        padded_indices = jnp.array(
            _pad(fwd_indices, max_accepted, 0),
            dtype=jnp.int32,
        )
        new_kv = _compact_kv_cache(
            fwd.updated_state,
            jnp.int32(cache_len),
            padded_indices,
            jnp.int32(num_accepted),
            max_accepted,
        )

        if num_accepted > 0:
            last_fwd_idx = int(fwd_indices[-1])
            new_lm = LMState(
                kv_cache=new_kv,
                hiddens=_extract_hiddens(fwd.activation_trace, 0, last_fwd_idx),
                logits=jnp.array(fwd.logits[0, last_fwd_idx]),
                position=cache_len + num_accepted,
                context=lm.context + tuple(accepted_tokens),
            )
        else:
            new_lm = lm

        # Bonus: sample on device
        bonus_seed = int(all_seeds_real[num_accepted])
        if self.use_gumbel:
            bonus_key = jax.random.key(bonus_seed & 0xFFFFFFFF)
            bonus_noise = jax.random.gumbel(
                bonus_key,
                (new_lm.logits.shape[-1],),
                dtype=jnp.float32,
            )
            bonus_token = int(
                jnp.argmax(
                    new_lm.logits.astype(jnp.float32) + bonus_noise,
                )
            )
        else:
            bonus_token = int(jnp.argmax(new_lm.logits))
        return accepted_tokens, (bonus_token, bonus_seed), new_lm


# ── Run (the only mutable object) ──────────────────────────────


class SpeculationRun:
    """A single speculative decoding session. Yields ``SpeculationStep``."""

    def __init__(
        self,
        ctx: SpeculationContext,
        prompt_ids: list[int],
        max_tokens: int,
        seed: int = 42,
    ) -> None:
        self.ctx = ctx
        self.max_tokens = max_tokens
        capacity = len(prompt_ids) + max_tokens + ctx.config.K * ctx.config.width + 16
        self.lm_state = ctx.prefill(prompt_ids, capacity)
        self.seed = seed
        self.result = SpeculativeDecodingResult()

    def _done(self) -> bool:
        return len(self.result.generated) >= self.max_tokens

    def _stopped(self) -> bool:
        return bool(self.result.generated and self.result.generated[-1] in self.ctx.eos_set)

    def __iter__(self) -> Iterator[SpeculationStep]:
        while not self._done() and not self._stopped():
            lm = self.lm_state

            trie = self.ctx.drafter.draft(lm, lm.context[-1], self.seed)
            accepted, (bonus, _bonus_seed), new_lm = self.ctx.verify(lm, trie)

            self.seed = _next_seed(self.seed, len(lm.context))

            for tok in accepted:
                self.result.generated.append(tok)

            new_lm = self.ctx.advance_one(new_lm, bonus)
            self.result.generated.append(bonus)

            self.lm_state = new_lm

            self.result.total_accepted += len(accepted)
            self.result.total_proposed += trie.total_nodes() - 1
            self.result.num_steps += 1

            yield SpeculationStep(
                accepted=accepted,
                bonus=bonus,
                proposed=trie.total_nodes() - 1,
                depth=trie.max_depth(),
            )
