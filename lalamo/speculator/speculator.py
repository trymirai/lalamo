from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import jax.numpy as jnp

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderResult
from lalamo.modules.token_mixers.state.kv_cache import compact_state_layers
from lalamo.speculator.proposal import AcceptedProposal, TrieProposal
from lalamo.speculator.sampler import GumbelSampler
from lalamo.speculator.state import LMState, RingBuffer, StateRequest, build_prefill_memory, commit_memory

__all__ = [
    "LMState",
    "RingBuffer",
    "SpeculationStep",
    "Speculator",
    "StateRequest",
]


@dataclass(frozen=True)
class SpeculationStep:
    token_ids: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    bonus_token_id: int


@dataclass(frozen=True)
class Speculator(ABC):
    decoder: Decoder
    sampler: GumbelSampler
    state_request: StateRequest = field(default_factory=StateRequest)

    @abstractmethod
    def draft(self, root: TrieProposal, state: LMState) -> TrieProposal: ...

    @abstractmethod
    def update(self, state: LMState, step: SpeculationStep) -> Self: ...

    def create_root_proposal(self, state: LMState) -> TrieProposal:
        return TrieProposal.create(
            root=state.root_bonus_id,
            root_sample_position=state.next_token_position + 1,
        )

    def prefill(self, prompt_ids: tuple[int, ...], max_output_length: int = 8192) -> tuple[Self, LMState]:
        capacity = len(prompt_ids) + max_output_length
        token_ids = jnp.array([prompt_ids], dtype=jnp.int32)
        token_positions = jnp.arange(len(prompt_ids), dtype=jnp.int32)[None, :]
        initial_state = self.decoder.init_static_state(batch_size=1, capacity=capacity)
        decoder_result = self.decoder(
            token_ids,
            token_positions,
            state=initial_state,
            return_updated_state=True,
            return_activation_trace=True,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        )
        root_bonus_logits = decoder_result.logits[0, -1]
        updated_state = decoder_result.updated_state
        assert updated_state is not None
        activation_trace = decoder_result.activation_trace
        assert activation_trace is not None
        memory = build_prefill_memory(self.state_request, activation_trace)
        return self, LMState(
            kv_cache=updated_state,
            next_token_position=len(prompt_ids),
            root_bonus_logits=root_bonus_logits,
            root_bonus_id=self.sampler.sample_logits(root_bonus_logits, len(prompt_ids)),
            memory=memory,
        )

    def step(self, state: LMState) -> tuple[Self, LMState, SpeculationStep]:
        proposal = self.draft(self.create_root_proposal(state), state)
        decoder_result, sampled_token_ids = proposal.forward(
            decoder=self.decoder,
            kv_cache=state.kv_cache,
            next_token_position=state.next_token_position,
            sampler=self.sampler,
            return_activation_trace=True,
        )
        accepted = proposal.verify(sampled_token_ids)
        next_state = self.build_next_state(state, decoder_result, proposal, accepted)
        step = SpeculationStep(
            token_ids=(state.root_bonus_id, *accepted.token_ids),
            accepted_token_ids=accepted.token_ids,
            bonus_token_id=accepted.bonus_token_id,
        )
        return self.update(state, step), next_state, step

    def build_next_state(
        self,
        state: LMState,
        decoder_result: DecoderResult,
        proposal: TrieProposal,
        accepted: AcceptedProposal,
    ) -> LMState:
        updated_state = decoder_result.updated_state
        assert updated_state is not None
        cache_length = state.kv_cache[0].prefix_lengths()
        if cache_length.ndim > 0:
            cache_length = cache_length[0]
        new_kv_cache = compact_state_layers(
            updated_state,
            cache_len=jnp.asarray(cache_length, dtype=jnp.int32),
            accepted_indices=accepted.compact_indices,
            num_accepted=jnp.array(accepted.num_compact_indices, dtype=jnp.int32),
            max_slots=len(proposal.nodes),
        )
        memory = commit_memory(
            self.state_request,
            state,
            decoder_result,
            accepted.compact_indices,
            accepted.num_compact_indices,
        )
        return LMState(
            kv_cache=new_kv_cache,
            next_token_position=state.next_token_position + accepted.num_compact_indices,
            root_bonus_logits=decoder_result.logits[0, accepted.terminal_node_index],
            root_bonus_id=accepted.bonus_token_id,
            memory=memory,
        )
