import dataclasses
from dataclasses import dataclass, field
from typing import Self

import jax.numpy as jnp
import numpy as np
from jaxtyping import Int, UInt

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import DecoderResult
from lalamo.modules.token_mixers.state.common import CompactableStateLayer
from lalamo.modules.token_mixers.state.kv_cache import compact_state_layers
from lalamo.speculator.common import LMState, SpeculationStep, Speculator
from lalamo.speculator.sampler import GumbelMaxSampler, GumbelSeed
from lalamo.speculator.utils import extract_activations, pad_or_trim


@dataclass(frozen=True)
class TreeVerifyResult:
    accepted_tokens: list[int]
    accepted_indices: Int[np.ndarray, " accepted"]
    all_seeds: UInt[np.ndarray, " nodes"]
    num_draft_nodes: int


@dataclass(frozen=True)
class FlatTrie:
    token_ids: Int[np.ndarray, " nodes"]
    seeds: UInt[np.ndarray, " nodes"]
    depths: Int[np.ndarray, " nodes"]
    parent_indices: Int[np.ndarray, " nodes"]

    @property
    def num_nodes(self) -> int:
        return len(self.token_ids)

    def positions(self, base_position: int) -> Int[np.ndarray, " nodes"]:
        return base_position + self.depths

    def accept(
        self,
        sampled_tokens: Int[np.ndarray, " nodes"],
    ) -> tuple[list[int], Int[np.ndarray, " accepted"]]:
        if self.num_nodes == 0:
            return [], np.array([], dtype=np.int32)

        children_map: dict[int, dict[int, int]] = {}
        for i in range(self.num_nodes):
            parent = int(self.parent_indices[i])
            children_map.setdefault(parent, {})[int(self.token_ids[i])] = i

        root_sampled = int(sampled_tokens[0])
        root_children = children_map.get(0, {})
        if root_sampled not in root_children:
            return [], np.array([], dtype=np.int32)

        child_idx = root_children[root_sampled]
        accepted_tokens = [root_sampled]
        accepted_indices = [child_idx]

        while child_idx in children_map:
            sampled = int(sampled_tokens[child_idx])
            siblings = children_map[child_idx]
            if sampled not in siblings:
                break
            child_idx = siblings[sampled]
            accepted_tokens.append(sampled)
            accepted_indices.append(child_idx)

        return accepted_tokens, np.array(accepted_indices, dtype=np.int32)


@dataclass(frozen=True, eq=False)
class TrieNode:
    token: int
    seed: int
    depth: int = 0
    children: list["TrieNode"] = field(default_factory=list)

    def add_child(self, token: int, seed: int) -> "TrieNode":
        child = TrieNode(token=token, seed=seed, depth=self.depth + 1)
        self.children.append(child)
        return child

    def get_child(self, token: int) -> "TrieNode | None":
        for child in self.children:
            if child.token == token:
                return child
        return None

    def total_nodes(self) -> int:
        return 1 + sum(child.total_nodes() for child in self.children)

    def max_depth(self) -> int:
        if not self.children:
            return self.depth
        return max(child.max_depth() for child in self.children)

    def linearize(self, *, include_root: bool = True) -> FlatTrie:
        token_ids: list[int] = []
        seeds: list[int] = []
        depths: list[int] = []
        parent_indices: list[int] = []

        stack: list[tuple[TrieNode, int]] = [(self, -1)]

        while stack:
            node, parent_idx = stack.pop()
            cur_idx = len(token_ids)
            token_ids.append(node.token)
            seeds.append(node.seed)
            depths.append(node.depth)
            parent_indices.append(parent_idx)
            stack.extend((child, cur_idx) for child in reversed(node.children))

        if not include_root:
            token_ids = token_ids[1:]
            seeds = seeds[1:]
            depths = [d - 1 for d in depths[1:]]
            parent_indices = [-1 if p == 0 else p - 1 for p in parent_indices[1:]]

        return FlatTrie(
            token_ids=np.array(token_ids, dtype=np.int32),
            seeds=np.array(seeds, dtype=np.uint64),
            depths=np.array(depths, dtype=np.int32),
            parent_indices=np.array(parent_indices, dtype=np.int32),
        )


@dataclass(frozen=True, kw_only=True)
class TreeSpeculator(Speculator[TrieNode]):
    seed: GumbelSeed

    @property
    def budget(self) -> int:
        return self.config.width * self.config.K

    def prefill(self, prompt_ids: list[int]) -> tuple[Self, LMState]:
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
        prefill_range = self.prefill_hidden_range
        prefill_positions = slice(None) if prefill_range is None else slice(-prefill_range, None)
        assert fwd.activation_trace is not None
        assert fwd.updated_state is not None
        layer_outputs, output_norm = extract_activations(
            fwd.activation_trace,
            sample_index=0,
            positions=prefill_positions,
            trace_layer_outputs=self.trace_layer_outputs,
            trace_output_norm=self.trace_output_norm,
        )
        lm_state = LMState(
            kv_cache=fwd.updated_state,
            layer_outputs=layer_outputs,
            output_norm=output_norm,
            logits=logits,
            position=len(prompt_ids),
            bonus=self.seed.derive(1).sample(logits),
        )
        return self, lm_state

    def step(self, lm: LMState) -> tuple[Self, LMState, SpeculationStep]:
        proposal = self.draft(lm)
        fwd, flat = self.tree_forward(lm, proposal)
        result = self.sample_and_accept(proposal, fwd, flat)
        new_self, new_lm = self.build_next_state(lm, fwd, result)
        return new_self, new_lm, SpeculationStep(accepted=result.accepted_tokens, bonus=new_lm.bonus)

    def tree_forward(self, lm: LMState, trie: TrieNode) -> tuple[DecoderResult, FlatTrie]:
        flat = trie.linearize(include_root=False)
        first_layer = lm.kv_cache[0]
        if not isinstance(first_layer, CompactableStateLayer):
            raise TypeError(
                f"speculative decoding requires CompactableStateLayer, got {type(first_layer).__name__}",
            )
        cache_len = first_layer.prefix_length_for_sample(0)
        max_fwd = 1 + max(self.budget, flat.num_nodes)

        tok_ids = jnp.array(
            pad_or_trim(np.concatenate([[lm.bonus], flat.token_ids]), max_fwd, 0)[None, :],
            dtype=jnp.int32,
        )
        positions = jnp.array(
            pad_or_trim(np.concatenate([[cache_len], flat.positions(cache_len + 1)]), max_fwd, cache_len)[None, :],
            dtype=jnp.int32,
        )
        shifted_parents = np.where(flat.parent_indices == -1, 0, flat.parent_indices + 1)
        parent_indices = jnp.array(
            pad_or_trim(np.concatenate([[-1], shifted_parents]), max_fwd, -1)[None, :],
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
        trie: TrieNode,
        fwd: DecoderResult,
        flat: FlatTrie,
    ) -> TreeVerifyResult:
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
        return TreeVerifyResult(accepted_tokens, accepted_indices, all_seeds, flat.num_nodes)

    def build_next_state(
        self,
        lm: LMState,
        fwd: DecoderResult,
        result: TreeVerifyResult,
    ) -> tuple[Self, LMState]:
        first_layer = lm.kv_cache[0]
        if not isinstance(first_layer, CompactableStateLayer):
            raise TypeError(
                f"speculative decoding requires CompactableStateLayer, got {type(first_layer).__name__}",
            )
        cache_len = first_layer.prefix_length_for_sample(0)
        num_accepted = len(result.accepted_tokens)

        kept_fwd = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                result.accepted_indices.astype(np.int32),
            ],
        )
        total_kept = 1 + num_accepted
        max_compact = 1 + max(self.budget, result.num_draft_nodes)

        assert fwd.updated_state is not None
        cache_len_arr = jnp.int32(cache_len)
        accepted_idx = jnp.array(pad_or_trim(kept_fwd, max_compact, 0), dtype=jnp.int32)
        num_acc = jnp.int32(total_kept)
        new_kv = compact_state_layers(fwd.updated_state, cache_len_arr, accepted_idx, num_acc, max_compact)

        last_fwd_idx = int(kept_fwd[total_kept - 1])
        last_logits = jnp.array(fwd.logits[0, last_fwd_idx])
        new_bonus = GumbelSeed(int(result.all_seeds[num_accepted])).sample(last_logits)

        assert fwd.activation_trace is not None
        layer_outputs, output_norm = extract_activations(
            fwd.activation_trace,
            sample_index=0,
            positions=jnp.array(kept_fwd[:total_kept], dtype=jnp.int32),
            trace_layer_outputs=self.trace_layer_outputs,
            trace_output_norm=self.trace_output_norm,
        )
        new_lm = LMState(
            kv_cache=new_kv,
            layer_outputs=layer_outputs,
            output_norm=output_norm,
            logits=last_logits,
            position=cache_len + total_kept,
            bonus=new_bonus,
        )
        new_self = dataclasses.replace(self, seed=self.seed.advance(lm.position))
        return new_self, new_lm
