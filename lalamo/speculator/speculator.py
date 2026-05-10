from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Literal, Self

from annotated_types import Ge
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderResult
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
    output_logits: bool = False
    layer_indices: tuple[int, ...] = ()
    context_length: Annotated[int, Ge(1)] | Literal["all"] = 1

    @property
    def context_slice(self) -> slice:
        if self.context_length == "all":
            return slice(None)
        return slice(-self.context_length, None)

    def prefill_compact_indices(self, prompt_length: int) -> Int[Array, " positions"]:
        start = 0
        if self.context_length != "all":
            start = max(prompt_length - self.context_length, 0)
        return jnp.arange(start, prompt_length, dtype=jnp.int32)


@dataclass(frozen=True)
class RequestedState:
    token_positions: Int[Array, " positions"] = field(default_factory=lambda: jnp.array((), dtype=jnp.int32))
    output_norm: Float[Array, "positions channels"] | None = None
    layer_outputs: tuple[Float[Array, "positions channels"], ...] = ()
    accepted_token_ids: Int[Array, " positions"] = field(default_factory=lambda: jnp.array((), dtype=jnp.int32))
    accepted_token_logits: Float[Array, "positions vocab"] | None = None


@dataclass(frozen=True)
class LMState:
    kv_cache: State
    next_token_position: int
    root_bonus_logits: Float[Array, " vocab"]
    root_bonus_id: int
    requested: RequestedState = field(default_factory=RequestedState)


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

    def prefill(self, prompt_ids: tuple[int, ...]) -> tuple[Self, LMState]:
        token_ids = jnp.array([prompt_ids], dtype=jnp.int32)
        token_positions = jnp.arange(len(prompt_ids), dtype=jnp.int32)[None, :]
        decoder_result = self.decoder(
            token_ids,
            token_positions,
            state=None,
            return_updated_state=True,
            return_activation_trace=True,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        )
        root_bonus_logits = decoder_result.logits[0, -1]
        updated_state = decoder_result.updated_state
        assert updated_state is not None
        compact_indices = self.state_request.prefill_compact_indices(len(prompt_ids))
        return self, LMState(
            kv_cache=updated_state,
            next_token_position=len(prompt_ids),
            root_bonus_logits=root_bonus_logits,
            root_bonus_id=self.sampler.sample_logits(root_bonus_logits, len(prompt_ids)),
            requested=self.collect_requested_state(
                previous=RequestedState(),
                decoder_result=decoder_result,
                compact_indices=compact_indices,
                num_compact_indices=compact_indices.shape[0],
            ),
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
        new_kv_cache = compact_state_layers(
            updated_state,
            cache_len=jnp.array(state.next_token_position, dtype=jnp.int32),
            accepted_indices=accepted.compact_indices,
            num_accepted=jnp.array(accepted.num_compact_indices, dtype=jnp.int32),
            max_slots=len(proposal.nodes),
        )
        return LMState(
            kv_cache=new_kv_cache,
            next_token_position=state.next_token_position + accepted.num_compact_indices,
            root_bonus_logits=decoder_result.logits[0, accepted.terminal_node_index],
            root_bonus_id=accepted.bonus_token_id,
            requested=self.collect_requested_state(
                previous=state.requested,
                decoder_result=decoder_result,
                compact_indices=accepted.compact_indices,
                num_compact_indices=accepted.num_compact_indices,
                num_accepted_tokens=len(accepted.token_ids),
            ),
        )

    def collect_requested_state(
        self,
        previous: RequestedState,
        decoder_result: DecoderResult,
        compact_indices: Int[Array, " max_slots"],
        num_compact_indices: int,
        num_accepted_tokens: int = 0,
    ) -> RequestedState:
        activation_trace = decoder_result.activation_trace
        assert activation_trace is not None
        context_slice = self.state_request.context_slice

        def compact_rows(values: Array, row_indices: Int[Array, " rows"], num_rows: int) -> Array:
            slots = jnp.arange(row_indices.shape[0], dtype=jnp.int32)
            row_indices = jnp.where(slots < num_rows, row_indices, 0)
            return values[row_indices][:num_rows]

        def append_context(previous_context: Array | None, current_context: Array) -> Array:
            context = (
                current_context
                if previous_context is None
                else jnp.concatenate([previous_context, current_context], axis=0)
            )
            return context[context_slice]

        token_positions = append_context(
            previous.token_positions,
            compact_rows(activation_trace.token_positions[0], compact_indices, num_compact_indices),
        )
        accepted_token_ids = append_context(
            previous.accepted_token_ids,
            compact_rows(activation_trace.token_ids[0], compact_indices[1:], num_accepted_tokens),
        )

        output_norm = None
        if self.state_request.output_norm:
            output_norm = append_context(
                previous.output_norm,
                compact_rows(activation_trace.output_norm[0], compact_indices, num_compact_indices),
            )

        layer_outputs = tuple(
            append_context(
                previous.layer_outputs[index] if index < len(previous.layer_outputs) else None,
                compact_rows(activation_trace.layer_results[layer_index].outputs[0], compact_indices, num_compact_indices),
            )
            for index, layer_index in enumerate(self.state_request.layer_indices)
        )

        accepted_token_logits = None
        if self.state_request.output_logits:
            accepted_token_logits = append_context(
                previous.accepted_token_logits,
                compact_rows(decoder_result.logits[0], compact_indices, num_accepted_tokens),
            )

        return RequestedState(
            token_positions=token_positions,
            output_norm=output_norm,
            layer_outputs=layer_outputs,
            accepted_token_ids=accepted_token_ids,
            accepted_token_logits=accepted_token_logits,
        )
