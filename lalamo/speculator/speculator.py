from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self, cast

import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderActivationTrace, DecoderResult
from lalamo.modules.token_mixers.state.common import State
from lalamo.modules.token_mixers.state.kv_cache import compact_state_layers
from lalamo.speculator.proposal import AcceptedProposal, TrieProposal
from lalamo.speculator.sampler import GumbelSampler

__all__ = [
    "LMState",
    "RequestedState",
    "SpeculationStep",
    "Speculator",
    "StateRequest",
]


@dataclass(frozen=True)
class StateRequest:
    output_norm: bool = False
    layer_indices: tuple[int, ...] = ()
    history_length: int | None = 1

    @property
    def activation_trace(self) -> bool:
        return self.output_norm or bool(self.layer_indices)


@dataclass(frozen=True)
class RequestedState:
    token_positions: tuple[int, ...]
    output_norm: Float[Array, "positions channels"] | None = None
    layer_indices: tuple[int, ...] = ()
    layer_outputs: tuple[Float[Array, "positions channels"], ...] = ()


@dataclass(frozen=True)
class LMState:
    kv_cache: State
    next_token_position: int
    logits: Float[Array, " vocab"]
    bonus_token_id: int
    requested: RequestedState = field(default_factory=lambda: RequestedState(token_positions=()))


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
    def draft(self, state: LMState) -> TrieProposal: ...

    @abstractmethod
    def update(self, state: LMState, step: SpeculationStep) -> Self: ...

    def step(self, state: LMState) -> tuple[Self, LMState, SpeculationStep]:
        proposal = self.draft(state)
        decoder_result = self.forward_proposal(state, proposal)
        sampled_token_ids = self.sample_proposal(decoder_result, proposal)
        accepted = proposal.verify(sampled_token_ids)
        next_state = self.build_next_state(state, decoder_result, proposal, accepted)
        step = SpeculationStep(
            token_ids=(state.bonus_token_id, *accepted.token_ids),
            accepted_token_ids=accepted.token_ids,
            bonus_token_id=accepted.bonus_token_id,
        )
        return self.update(state, step), next_state, step

    def forward_proposal(self, state: LMState, proposal: TrieProposal) -> DecoderResult:
        token_ids = jnp.array([proposal.token_ids], dtype=jnp.int32)
        token_positions = jnp.array(
            [[state.next_token_position + depth for depth in proposal.depths]],
            dtype=jnp.int32,
        )
        parent_indices = jnp.array([proposal.parent_indices], dtype=jnp.int32)

        return self.decoder(
            token_ids,
            token_positions,
            state.kv_cache,
            return_updated_state=True,
            return_activation_trace=self.state_request.activation_trace,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            attention_parent_indices=parent_indices,
        )

    def sample_proposal(self, decoder_result: DecoderResult, proposal: TrieProposal) -> tuple[int, ...]:
        return tuple(
            self.sampler.sample_logits(logits, sample_position)
            for logits, sample_position in zip(decoder_result.logits[0], proposal.sample_positions, strict=True)
        )

    def build_next_state(
        self,
        state: LMState,
        decoder_result: DecoderResult,
        proposal: TrieProposal,
        accepted: AcceptedProposal,
    ) -> LMState:
        new_kv_cache = compact_state_layers(
            cast("State", decoder_result.updated_state),
            cache_len=jnp.array(state.next_token_position, dtype=jnp.int32),
            accepted_indices=jnp.array(accepted.compact_indices, dtype=jnp.int32),
            num_accepted=jnp.array(accepted.num_compact_indices, dtype=jnp.int32),
            max_slots=len(proposal.nodes),
        )
        return LMState(
            kv_cache=new_kv_cache,
            next_token_position=state.next_token_position + accepted.num_compact_indices,
            logits=decoder_result.logits[0, accepted.terminal_node_index],
            bonus_token_id=accepted.bonus_token_id,
            requested=self.build_requested_state(state, decoder_result, proposal, accepted),
        )

    def build_requested_state(
        self,
        state: LMState,
        decoder_result: DecoderResult,
        proposal: TrieProposal,
        accepted: AcceptedProposal,
    ) -> RequestedState:
        if not self.state_request.activation_trace:
            return RequestedState(token_positions=())

        compact_indices = accepted.compact_indices[: accepted.num_compact_indices]
        token_positions = tuple(state.next_token_position + proposal.depths[index] for index in compact_indices)
        token_positions = (*state.requested.token_positions, *token_positions)
        if self.state_request.history_length is not None:
            token_positions = token_positions[-self.state_request.history_length :]

        activation_trace = cast("DecoderActivationTrace", decoder_result.activation_trace)
        jax_compact_indices = jnp.array(compact_indices, dtype=jnp.int32)
        output_norm = None
        if self.state_request.output_norm:
            output_norm = activation_trace.output_norm[0, jax_compact_indices]
            if state.requested.output_norm is not None:
                output_norm = jnp.concatenate([state.requested.output_norm, output_norm], axis=0)
            if self.state_request.history_length is not None:
                output_norm = output_norm[-self.state_request.history_length :]

        layer_outputs = tuple(
            activation_trace.layer_results[layer_index].outputs[0, jax_compact_indices]
            for layer_index in self.state_request.layer_indices
        )
        if state.requested.layer_outputs:
            layer_outputs = tuple(
                jnp.concatenate([previous, current], axis=0)
                for previous, current in zip(state.requested.layer_outputs, layer_outputs, strict=True)
            )
        if self.state_request.history_length is not None:
            layer_outputs = tuple(output[-self.state_request.history_length :] for output in layer_outputs)

        return RequestedState(
            token_positions=token_positions,
            output_norm=output_norm,
            layer_indices=self.state_request.layer_indices,
            layer_outputs=layer_outputs,
        )
