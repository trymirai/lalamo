import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Self

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, UInt

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderResult
from lalamo.modules.token_mixers.state.common import State
from lalamo.modules.token_mixers.state.kv_cache import StaticKVCacheLayer
from lalamo.speculator.drafter import Drafter
from lalamo.speculator.sampler import GumbelMaxSampler, GumbelSeed
from lalamo.speculator.trie import FlatTrie, TrieNode
from lalamo.speculator.utils import extract_activations, pad_or_trim


@dataclass(frozen=True)
class LMState:
    """Immutable snapshot after the last verified position.

    Invariants:
    - ``logits`` is the next-token distribution at the head position.
    - ``bonus`` is sampled from ``logits`` and is always present.
    - The next verify batch starts with ``bonus`` as its first token.
    - ``bonus`` is NOT yet materialized in the KV cache.
    - ``layer_outputs`` has one ``(suffix, d)`` array per layer in
      ``drafter.trace_layer_outputs``. After prefill ``suffix`` equals
      ``prompt_len`` (or ``drafter.prefill_hidden_range`` if specified);
      after verify ``suffix == 1 + num_accepted`` (bonus + accepted rows).
    - ``output_norm`` is populated iff ``drafter.trace_output_norm`` is True;
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


@dataclass(frozen=True)
class SpeculationStep:
    accepted: list[int]
    bonus: int


@dataclass
class SpeculativeDecodingResult:
    num_steps: int = 0
    total_accepted: int = 0
    generated: list[int] = field(default_factory=list)

    @property
    def mean_draft_accepted(self) -> float:
        return self.total_accepted / max(self.num_steps, 1)

    @property
    def speculation_rate(self) -> float:
        return self.total_accepted / max(len(self.generated), 1)

    @property
    def tokens_per_step(self) -> float:
        return len(self.generated) / max(self.num_steps, 1)


@dataclass(frozen=True)
class VerifyResult:
    accepted_tokens: list[int]
    accepted_indices: Int[np.ndarray, " accepted"]
    all_seeds: UInt[np.ndarray, " nodes"]
    num_draft_nodes: int


@dataclass(frozen=True)
class SpeculationContext:
    decoder: Decoder
    drafter: Drafter
    config: SamplerConfig
    eos_set: frozenset[int]

    @classmethod
    def create(
        cls,
        decoder: Decoder,
        drafter: Drafter,
        config: SamplerConfig,
        eos_set: set[int],
    ) -> Self:
        return cls(
            decoder=decoder,
            drafter=drafter,
            config=config,
            eos_set=frozenset(eos_set),
        )

    @property
    def generation_capacity(self) -> int:
        return self.config.max_tokens + self.config.K * self.config.width + 16

    def prefill(self, prompt_ids: list[int], seed: GumbelSeed) -> LMState:
        state = self.decoder.init_static_state(1, self.generation_capacity + len(prompt_ids))
        prefix = jnp.array([prompt_ids], dtype=jnp.int32)
        fwd = self.decoder(
            prefix,
            jnp.arange(len(prompt_ids))[None, :],
            state,
            return_updated_state=True,
            return_activation_trace=True,
        )
        logits = fwd.logits[0, -1]
        prefill_range = self.drafter.prefill_hidden_range
        prefill_positions = slice(None) if prefill_range is None else slice(-prefill_range, None)
        assert fwd.activation_trace is not None
        assert fwd.updated_state is not None
        layer_outputs, output_norm = extract_activations(
            fwd.activation_trace,
            sample_index=0,
            positions=prefill_positions,
            trace_layer_outputs=self.drafter.trace_layer_outputs,
            trace_output_norm=self.drafter.trace_output_norm,
        )
        return LMState(
            kv_cache=fwd.updated_state,
            layer_outputs=layer_outputs,
            output_norm=output_norm,
            logits=logits,
            position=len(prompt_ids),
            bonus=seed.sample(logits),
        )

    @property
    def budget(self) -> int:
        return self.config.width * self.config.K

    def tree_forward(self, lm: LMState, trie: TrieNode) -> tuple[DecoderResult, FlatTrie]:
        flat = trie.linearize(include_root=False)
        first_layer = lm.kv_cache[0]
        assert isinstance(first_layer, StaticKVCacheLayer)
        cache_len = int(first_layer.current_length[0])
        max_fwd = 1 + max(self.budget, flat.num_nodes)

        tok_ids = jnp.array(
            pad_or_trim(np.concatenate([[lm.bonus], flat.token_ids]), max_fwd, 0)[None, :],
            dtype=jnp.int32,
        )
        positions = jnp.array(
            pad_or_trim(np.concatenate([[cache_len], flat.positions(cache_len + 1)]), max_fwd, cache_len)[None, :],
            dtype=jnp.int32,
        )
        parent_indices = jnp.array(
            pad_or_trim(np.concatenate([[-1], flat.parent_indices]), max_fwd, -1)[None, :],
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
        )
        return fwd, flat

    def sample_and_accept(
        self,
        lm: LMState,  # noqa: ARG002
        trie: TrieNode,
        fwd: DecoderResult,
        flat: FlatTrie,
    ) -> VerifyResult:
        max_fwd = 1 + max(self.budget, flat.num_nodes)

        root_logits = fwd.logits[0, 0].astype(jnp.float32)
        all_logits = jnp.concatenate([root_logits[None, :], fwd.logits[0, 1:]], axis=0).astype(jnp.float32)
        all_seeds = np.concatenate([np.array([trie.seed], dtype=np.uint64), flat.seeds])
        padded_seeds = pad_or_trim(all_seeds, max_fwd, 0)

        sampler = GumbelMaxSampler()
        sampled_all = np.asarray(
            sampler.sample(all_logits, jnp.array(padded_seeds & 0xFFFFFFFF, dtype=jnp.uint32)),
        )

        full_flat = FlatTrie(
            token_ids=np.concatenate([[trie.token], flat.token_ids]),
            seeds=all_seeds,
            depths=np.concatenate([[0], flat.depths + 1]),
            parent_indices=np.concatenate([[-1], flat.parent_indices + 1]),
        )
        accepted_tokens, accepted_indices = full_flat.accept(sampled_all[: flat.num_nodes + 1])
        return VerifyResult(accepted_tokens, accepted_indices, all_seeds, flat.num_nodes)

    def build_next_state(
        self,
        lm: LMState,
        fwd: DecoderResult,
        result: VerifyResult,
    ) -> LMState:
        first_layer = lm.kv_cache[0]
        assert isinstance(first_layer, StaticKVCacheLayer)
        cache_len = int(first_layer.current_length[0])
        num_accepted = len(result.accepted_tokens)

        kept_fwd = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                result.accepted_indices.astype(np.int32),
            ]
        )
        total_kept = 1 + num_accepted
        max_compact = 1 + max(self.budget, result.num_draft_nodes)

        assert fwd.updated_state is not None
        cache_len_arr = jnp.int32(cache_len)
        accepted_idx = jnp.array(pad_or_trim(kept_fwd, max_compact, 0), dtype=jnp.int32)
        num_acc = jnp.int32(total_kept)
        new_layers: list[StaticKVCacheLayer] = []
        for layer in fwd.updated_state:
            if not isinstance(layer, StaticKVCacheLayer):
                raise TypeError(
                    f"speculative decoding requires StaticKVCacheLayer, got {type(layer).__name__}",
                )
            new_layers.append(layer.compact(cache_len_arr, accepted_idx, num_acc, max_compact))
        new_kv = State(new_layers)

        last_fwd_idx = int(kept_fwd[total_kept - 1])
        last_logits = jnp.array(fwd.logits[0, last_fwd_idx])
        new_bonus = GumbelSeed(int(result.all_seeds[num_accepted])).sample(last_logits)

        assert fwd.activation_trace is not None
        layer_outputs, output_norm = extract_activations(
            fwd.activation_trace,
            sample_index=0,
            positions=kept_fwd[:total_kept],
            trace_layer_outputs=self.drafter.trace_layer_outputs,
            trace_output_norm=self.drafter.trace_output_norm,
        )
        return LMState(
            kv_cache=new_kv,
            layer_outputs=layer_outputs,
            output_norm=output_norm,
            logits=last_logits,
            position=cache_len + total_kept,
            bonus=new_bonus,
        )

    def verify(self, lm: LMState, trie: TrieNode) -> tuple[list[int], LMState]:
        fwd, flat = self.tree_forward(lm, trie)
        result = self.sample_and_accept(lm, trie, fwd, flat)
        return result.accepted_tokens, self.build_next_state(lm, fwd, result)


class SpeculationRun:
    def __init__(
        self,
        ctx: SpeculationContext,
        prompt_ids: list[int],
        seed: int = 42,
    ) -> None:
        self.seed = GumbelSeed(seed)
        self.lm_state = ctx.prefill(prompt_ids, self.seed.advance(0))
        self.ctx = dataclasses.replace(ctx, drafter=ctx.drafter.on_prefill(self.lm_state))
        self.result = SpeculativeDecodingResult()

    def done(self) -> bool:
        return len(self.result.generated) >= self.ctx.config.max_tokens

    def stopped(self) -> bool:
        return self.lm_state.bonus in self.ctx.eos_set or (
            bool(self.result.generated and self.result.generated[-1] in self.ctx.eos_set)
        )

    def advance(self, lm: LMState, accepted: list[int], new_lm: LMState) -> SpeculationStep:
        self.ctx = dataclasses.replace(
            self.ctx,
            drafter=self.ctx.drafter.update_after_verify(lm, accepted, new_lm.bonus, new_lm),
        )
        self.seed = self.seed.advance(lm.position)

        emitted = [lm.bonus, *accepted]
        remaining = self.ctx.config.max_tokens - len(self.result.generated)
        emitted = emitted[:remaining]
        self.result.generated.extend(emitted)

        self.lm_state = new_lm
        self.result.total_accepted += len(accepted)
        self.result.num_steps += 1
        return SpeculationStep(accepted=accepted, bonus=new_lm.bonus)

    def __iter__(self) -> Iterator[SpeculationStep]:
        while not self.done() and not self.stopped():
            lm = self.lm_state
            proposal = self.ctx.drafter.draft(lm, self.seed.value)
            accepted, new_lm = self.ctx.verify(lm, proposal)
            yield self.advance(lm, accepted, new_lm)

        # Emit the un-emitted trailing EOS bonus (advance() emits the previous bonus, not new_lm's).
        if self.lm_state.bonus in self.ctx.eos_set:
            self.result.generated.append(self.lm_state.bonus)
