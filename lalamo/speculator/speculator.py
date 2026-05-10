from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, Int

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderResult
from lalamo.modules.token_mixers.state.kv_cache import compact_state_layers
from lalamo.speculator.proposal import AcceptedProposal, TrieProposal
from lalamo.speculator.sampler import GumbelSampler
from lalamo.speculator.state import LMState, MemoryBuffers, RingBuffer, StateRequest

__all__ = [
    "LMState",
    "MemoryBuffers",
    "RingBuffer",
    "Speculator",
    "StateRequest",
]


@dataclass(frozen=True)
class Speculator(ABC):
    decoder: Decoder
    sampler: GumbelSampler
    capacity: int = 8192
    state_request: StateRequest = field(default_factory=StateRequest)

    @abstractmethod
    def draft(self, state: LMState) -> TrieProposal: ...

    @abstractmethod
    def update(self, state: LMState, accepted: AcceptedProposal) -> Self: ...

    # TODO: Remove prefill/step/build_next_state into somewhere else
    # And add utils on AcceptedProposal for measuring mal etc is clean ig
    def prefill(
        self,
        prompt_ids: Int[Array, "batch prompt"],
        prompt_lengths: Int[Array, " batch"],
    ) -> tuple[Self, LMState]:
        batch_size, prompt_capacity = prompt_ids.shape
        token_positions = jnp.broadcast_to(
            jnp.arange(prompt_capacity, dtype=jnp.int32)[None, :],
            prompt_ids.shape,
        )
        initial_state = self.decoder.init_static_state(
            batch_size=batch_size,
            capacity=self.capacity,
        )
        decoder_result = self.decoder(
            prompt_ids,
            token_positions,
            state=initial_state,
            return_updated_state=True,
            return_activation_trace=True,
            lengths_without_padding=prompt_lengths,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        )
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        root_bonus_logits = decoder_result.logits[batch_indices, prompt_lengths - 1]
        updated_state = decoder_result.updated_state
        assert updated_state is not None
        activation_trace = decoder_result.activation_trace
        assert activation_trace is not None
        memory = MemoryBuffers.from_prefill(self.state_request, activation_trace, prompt_lengths)
        return self, LMState(
            kv_cache=updated_state,
            next_token_position=prompt_lengths,
            root_bonus_id=self.sampler.sample_logits(
                root_bonus_logits,
                prompt_lengths,
            ),
            memory=memory,
        )

    def step(self, state: LMState) -> tuple[Self, LMState, AcceptedProposal]:
        proposal = self.draft(state)
        decoder_result, sampled_token_ids = proposal.forward(
            decoder=self.decoder,
            kv_cache=state.kv_cache,
            next_token_positions=state.next_token_position,
            sampler=self.sampler,
            return_activation_trace=True,
        )
        accepted = proposal.verify(sampled_token_ids)
        next_state = self.build_next_state(state, decoder_result, proposal, accepted)
        return self.update(state, accepted), next_state, accepted

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
        new_kv_cache = compact_state_layers(
            updated_state,
            cache_len=cache_length,
            accepted_indices=accepted.compact_indices,
            num_accepted=accepted.num_compact_indices,
            max_slots=proposal.budget,
        )
        memory = state.memory.commit(
            self.state_request,
            decoder_result,
            accepted.compact_indices,
            accepted.num_compact_indices,
        )
        return LMState(
            kv_cache=new_kv_cache,
            next_token_position=state.next_token_position + accepted.num_compact_indices,
            root_bonus_id=accepted.bonus_token_ids,
            memory=memory,
        )
