from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

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

    def prefill(self, prompt_ids: tuple[int, ...]) -> tuple[Self, LMState]:
        token_ids = jnp.array([prompt_ids], dtype=jnp.int32)
        token_positions = jnp.arange(len(prompt_ids), dtype=jnp.int32)[None, :]
        decoder_result = self.decoder(
            token_ids,
            token_positions,
            state=None,
            return_updated_state=True,
            return_activation_trace=self.state_request.activation_trace,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        )
        logits = decoder_result.logits[0, -1]
        updated_state = decoder_result.updated_state
        assert updated_state is not None
        return self, LMState(
            kv_cache=updated_state,
            next_token_position=len(prompt_ids),
            logits=logits,
            bonus_token_id=self.sampler.sample_logits(logits, 0),
            requested=self.collect_requested_state(
                previous=RequestedState(token_positions=()),
                decoder_result=decoder_result,
                token_positions=self.prefill_trace_indices(len(prompt_ids)),
                trace_indices=self.prefill_trace_indices(len(prompt_ids)),
            ),
        )

    def step(self, state: LMState) -> tuple[Self, LMState, SpeculationStep]:
        proposal = self.draft(state)
        decoder_result, sampled_token_ids = proposal.forward(
            decoder=self.decoder,
            kv_cache=state.kv_cache,
            next_token_position=state.next_token_position,
            sampler=self.sampler,
            return_activation_trace=self.state_request.activation_trace,
        )
        accepted = proposal.verify(sampled_token_ids)
        next_state = self.build_next_state(state, decoder_result, proposal, accepted)
        step = SpeculationStep(
            token_ids=(state.bonus_token_id, *accepted.token_ids),
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
        new_kv_cache = compact_state_layers(
            updated_state,
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
            requested=self.collect_requested_state(
                previous=state.requested,
                decoder_result=decoder_result,
                token_positions=tuple(
                    state.next_token_position + proposal.depths[index]
                    for index in accepted.compact_indices[: accepted.num_compact_indices]
                ),
                trace_indices=accepted.compact_indices[: accepted.num_compact_indices],
            ),
        )

    def prefill_trace_indices(self, prompt_length: int) -> tuple[int, ...]:
        start = 0
        if self.state_request.history_length is not None:
            start = max(prompt_length - self.state_request.history_length, 0)
        return tuple(range(start, prompt_length))

    def collect_requested_state(
        self,
        previous: RequestedState,
        decoder_result: DecoderResult,
        token_positions: tuple[int, ...],
        trace_indices: tuple[int, ...],
    ) -> RequestedState:
        if not self.state_request.activation_trace:
            return RequestedState(token_positions=())

        token_positions = (*previous.token_positions, *token_positions)
        if self.state_request.history_length is not None:
            token_positions = token_positions[-self.state_request.history_length :]

        activation_trace = decoder_result.activation_trace
        assert activation_trace is not None
        jax_trace_indices = jnp.array(trace_indices, dtype=jnp.int32)
        output_norm = None
        if self.state_request.output_norm:
            output_norm = activation_trace.output_norm[0, jax_trace_indices]
            if previous.output_norm is not None:
                output_norm = jnp.concatenate([previous.output_norm, output_norm], axis=0)
            if self.state_request.history_length is not None:
                output_norm = output_norm[-self.state_request.history_length :]

        layer_outputs = tuple(
            activation_trace.layer_results[layer_index].outputs[0, jax_trace_indices]
            for layer_index in self.state_request.layer_indices
        )
        if previous.layer_outputs:
            layer_outputs = tuple(
                jnp.concatenate([previous_output, new_output], axis=0)
                for previous_output, new_output in zip(previous.layer_outputs, layer_outputs, strict=True)
            )
        if self.state_request.history_length is not None:
            layer_outputs = tuple(output[-self.state_request.history_length :] for output in layer_outputs)

        return RequestedState(
            token_positions=token_positions,
            output_norm=output_norm,
            layer_indices=self.state_request.layer_indices,
            layer_outputs=layer_outputs,
        )
