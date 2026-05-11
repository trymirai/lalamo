from typing import Annotated

from annotated_types import Ge
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.modules.decoder import DecoderActivationTrace, DecoderResult
from lalamo.modules.token_mixers.state.common import State
from lalamo.modules.token_mixers.state.kv_cache import compact_state_layers
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.proposal import AcceptedProposal, TrieProposal

__all__ = [
    "LMState",
    "MemoryBuffers",
    "RingBuffer",
    "StateRequest",
]


class StateRequest(eqx.Module):
    token_id_capacity: Annotated[int, Ge(0)] = eqx.field(static=True, default=0)
    sampling_top_k_capacity: Annotated[int, Ge(0)] = eqx.field(static=True, default=0)
    sampling_top_k_count: Annotated[int, Ge(0)] = eqx.field(static=True, default=0)
    trace_top_k_capacity: Annotated[int, Ge(0)] = eqx.field(static=True, default=0)
    trace_top_k_count: Annotated[int, Ge(0)] = eqx.field(static=True, default=0)
    logsumexp_capacity: Annotated[int, Ge(0)] = eqx.field(static=True, default=0)
    output_norm_capacity: Annotated[int, Ge(0)] = eqx.field(static=True, default=0)
    layer_capacity: Annotated[int, Ge(0)] = eqx.field(static=True, default=0)
    layer_indices: tuple[int, ...] = eqx.field(static=True, default=())

    def __post_init__(self) -> None:
        assert (self.sampling_top_k_capacity > 0) == (self.sampling_top_k_count > 0), (
            "sampling_top_k_capacity and sampling_top_k_count must both be set or both be zero."
        )
        assert (self.trace_top_k_capacity > 0) == (self.trace_top_k_count > 0), (
            "trace_top_k_capacity and trace_top_k_count must both be set or both be zero."
        )
        assert (self.layer_capacity > 0) == (self.layer_indices != ()), (
            "layer_capacity and layer_indices must both be set or both be empty."
        )

    @property
    def requires_activation_trace(self) -> bool:
        return self.output_norm_capacity > 0 or self.layer_capacity > 0


class RingBuffer(eqx.Module):
    values: Array
    length: Int[Array, " batch"]

    @classmethod
    def empty(cls, batch_size: int, capacity: int, row_shape: tuple[int, ...], dtype: jnp.dtype) -> "RingBuffer":
        return cls(
            values=jnp.zeros((batch_size, capacity, *row_shape), dtype=dtype),
            length=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    @classmethod
    def from_rows(cls, rows: Array, count: Int[Array, " batch"], capacity: int) -> "RingBuffer":
        return cls.empty(rows.shape[0], capacity, rows.shape[2:], rows.dtype).append(rows, count)

    def append(self, rows: Array, count: Int[Array, " batch"]) -> "RingBuffer":
        count = jnp.asarray(count, dtype=jnp.int32)
        assert count.shape == (rows.shape[0],), "count must have shape (batch,)."
        assert self.values.shape[0] == rows.shape[0], "rows batch size must match buffer batch size."
        capacity = self.values.shape[1]
        slots = jnp.arange(capacity, dtype=jnp.int32)[None, :]
        write_count = jnp.minimum(count[:, None], capacity)
        write_slots = slots < write_count
        padded_rows = jnp.concatenate([jnp.zeros((rows.shape[0], 1, *rows.shape[2:]), dtype=rows.dtype), rows], axis=1)
        source_indices = jnp.where(write_slots, count[:, None] - write_count + slots + 1, 0)
        positions = (self.length[:, None] + count[:, None] - write_count + slots) % capacity
        batch_indices = jnp.arange(rows.shape[0], dtype=jnp.int32)[:, None]
        retained_rows = padded_rows[batch_indices, source_indices]
        current_rows = self.values[batch_indices, positions]
        write_mask = write_slots.reshape((rows.shape[0], capacity, *([1] * (rows.ndim - 2))))
        values = self.values.at[batch_indices, positions].set(jnp.where(write_mask, retained_rows, current_rows))
        return RingBuffer(values, self.length + count)

    def recent(self, length: int) -> tuple[Array, Bool[Array, "batch length"]]:
        retained_length = jnp.minimum(
            jnp.minimum(jnp.asarray(length, dtype=jnp.int32), self.length),
            self.values.shape[1],
        )
        start = self.length[:, None] - retained_length[:, None]
        slots = jnp.arange(length, dtype=jnp.int32)[None, :]
        positions = (slots + start) % self.values.shape[1]
        batch_indices = jnp.arange(self.values.shape[0], dtype=jnp.int32)[:, None]
        return self.values[batch_indices, positions], slots < retained_length[:, None]

    def window(self, start: Int[Array, " batch"], length: int) -> tuple[Array, Bool[Array, "batch length"]]:
        start = jnp.asarray(start, dtype=jnp.int32)
        slots = jnp.arange(length, dtype=jnp.int32)[None, :]
        absolute_positions = start[:, None] + slots
        retained_start = self.length - jnp.minimum(self.length, self.values.shape[1])
        valid = jnp.logical_and(
            absolute_positions >= retained_start[:, None],
            absolute_positions < self.length[:, None],
        )
        positions = absolute_positions % self.values.shape[1]
        batch_indices = jnp.arange(self.values.shape[0], dtype=jnp.int32)[:, None]
        return self.values[batch_indices, positions], valid


class MemoryBuffers(eqx.Module):
    token_ids: RingBuffer | None = None
    sampling_top_k_ids: RingBuffer | None = None
    sampling_top_k_logits: RingBuffer | None = None
    trace_top_k_ids: RingBuffer | None = None
    trace_top_k_logits: RingBuffer | None = None
    logsumexp: RingBuffer | None = None
    output_norm: RingBuffer | None = None
    layer_outputs: tuple[RingBuffer, ...] = ()

    @classmethod
    def empty(cls, state_request: StateRequest, batch_size: int, hidden_dim: int) -> "MemoryBuffers":
        token_ids = None
        if state_request.token_id_capacity > 0:
            token_ids = RingBuffer.empty(batch_size, state_request.token_id_capacity, (), jnp.int32)

        sampling_top_k_ids = None
        sampling_top_k_logits = None
        if state_request.sampling_top_k_capacity > 0:
            sampling_shape = (state_request.sampling_top_k_count,)
            sampling_top_k_ids = RingBuffer.empty(
                batch_size,
                state_request.sampling_top_k_capacity,
                sampling_shape,
                jnp.int32,
            )
            sampling_top_k_logits = RingBuffer.empty(
                batch_size,
                state_request.sampling_top_k_capacity,
                sampling_shape,
                jnp.float32,
            )

        trace_top_k_ids = None
        trace_top_k_logits = None
        if state_request.trace_top_k_capacity > 0:
            trace_shape = (state_request.trace_top_k_count,)
            trace_top_k_ids = RingBuffer.empty(batch_size, state_request.trace_top_k_capacity, trace_shape, jnp.int32)
            trace_top_k_logits = RingBuffer.empty(
                batch_size,
                state_request.trace_top_k_capacity,
                trace_shape,
                jnp.bfloat16,
            )

        logsumexp = None
        if state_request.logsumexp_capacity > 0:
            logsumexp = RingBuffer.empty(batch_size, state_request.logsumexp_capacity, (), jnp.float32)

        output_norm = None
        if state_request.output_norm_capacity > 0:
            output_norm = RingBuffer.empty(batch_size, state_request.output_norm_capacity, (hidden_dim,), jnp.bfloat16)

        layer_outputs = tuple(
            RingBuffer.empty(batch_size, state_request.layer_capacity, (hidden_dim,), jnp.bfloat16)
            for _ in state_request.layer_indices
        )

        return MemoryBuffers(
            token_ids=token_ids,
            sampling_top_k_ids=sampling_top_k_ids,
            sampling_top_k_logits=sampling_top_k_logits,
            trace_top_k_ids=trace_top_k_ids,
            trace_top_k_logits=trace_top_k_logits,
            logsumexp=logsumexp,
            output_norm=output_norm,
            layer_outputs=layer_outputs,
        )

    def append_prefill_chunk(
        self,
        state_request: StateRequest,
        token_ids: Int[Array, "batch chunk"],
        activation_trace: DecoderActivationTrace | None,
        chunk_lengths: Int[Array, " batch"],
    ) -> "MemoryBuffers":
        next_token_ids = self.token_ids
        if next_token_ids is not None:
            next_token_ids = next_token_ids.append(token_ids, chunk_lengths)

        output_norm = self.output_norm
        if output_norm is not None:
            assert activation_trace is not None
            output_norm = output_norm.append(activation_trace.output_norm.astype(jnp.bfloat16), chunk_lengths)

        layer_outputs = self.layer_outputs
        if layer_outputs:
            assert activation_trace is not None
            layer_outputs = tuple(
                previous_layer_output.append(
                    activation_trace.layer_results[layer_index].outputs.astype(jnp.bfloat16),
                    chunk_lengths,
                )
                for previous_layer_output, layer_index in zip(
                    layer_outputs,
                    state_request.layer_indices,
                    strict=True,
                )
            )

        return MemoryBuffers(
            token_ids=next_token_ids,
            sampling_top_k_ids=self.sampling_top_k_ids,
            sampling_top_k_logits=self.sampling_top_k_logits,
            trace_top_k_ids=self.trace_top_k_ids,
            trace_top_k_logits=self.trace_top_k_logits,
            logsumexp=self.logsumexp,
            output_norm=output_norm,
            layer_outputs=layer_outputs,
        )

    def commit(
        self,
        state_request: StateRequest,
        decoder_result: DecoderResult,
        compact_indices: Int[Array, "batch max_slots"],
        num_compact_indices: Int[Array, " batch"],
        emitted_token_ids: Int[Array, "batch max_slots"],
        sampling_top_k_ids: Int[Array, "batch max_slots top_k"] | None = None,
        sampling_top_k_logits: Float[Array, "batch max_slots top_k"] | None = None,
        trace_top_k_ids: Int[Array, "batch max_slots top_k"] | None = None,
        trace_top_k_logits: Float[Array, "batch max_slots top_k"] | None = None,
        logsumexp: Float[Array, "batch max_slots"] | None = None,
    ) -> "MemoryBuffers":
        activation_trace = decoder_result.activation_trace

        def compact_rows(values: Array) -> Array:
            slots = jnp.arange(compact_indices.shape[1], dtype=jnp.int32)[None, :]
            row_indices = jnp.where(slots < num_compact_indices[:, None], compact_indices, 0)
            batch_indices = jnp.arange(compact_indices.shape[0], dtype=jnp.int32)[:, None]
            return values[batch_indices, row_indices]

        token_ids = self.token_ids
        if token_ids is not None:
            token_ids = token_ids.append(emitted_token_ids, num_compact_indices)

        next_sampling_top_k_ids = self.sampling_top_k_ids
        if next_sampling_top_k_ids is not None:
            assert sampling_top_k_ids is not None
            next_sampling_top_k_ids = next_sampling_top_k_ids.append(sampling_top_k_ids, num_compact_indices)

        next_sampling_top_k_logits = self.sampling_top_k_logits
        if next_sampling_top_k_logits is not None:
            assert sampling_top_k_logits is not None
            next_sampling_top_k_logits = next_sampling_top_k_logits.append(sampling_top_k_logits, num_compact_indices)

        next_trace_top_k_ids = self.trace_top_k_ids
        if next_trace_top_k_ids is not None:
            assert trace_top_k_ids is not None
            next_trace_top_k_ids = next_trace_top_k_ids.append(trace_top_k_ids, num_compact_indices)

        next_trace_top_k_logits = self.trace_top_k_logits
        if next_trace_top_k_logits is not None:
            assert trace_top_k_logits is not None
            next_trace_top_k_logits = next_trace_top_k_logits.append(
                trace_top_k_logits.astype(jnp.bfloat16),
                num_compact_indices,
            )

        next_logsumexp = self.logsumexp
        if next_logsumexp is not None:
            assert logsumexp is not None
            next_logsumexp = next_logsumexp.append(logsumexp, num_compact_indices)

        output_norm = self.output_norm
        if output_norm is not None:
            assert activation_trace is not None
            output_norm = output_norm.append(
                compact_rows(activation_trace.output_norm).astype(jnp.bfloat16),
                num_compact_indices,
            )

        layer_outputs = self.layer_outputs
        if layer_outputs:
            assert activation_trace is not None
            layer_outputs = tuple(
                previous_layer_output.append(
                    compact_rows(activation_trace.layer_results[layer_index].outputs).astype(jnp.bfloat16),
                    num_compact_indices,
                )
                for previous_layer_output, layer_index in zip(
                    layer_outputs,
                    state_request.layer_indices,
                    strict=True,
                )
            )

        return MemoryBuffers(
            token_ids=token_ids,
            sampling_top_k_ids=next_sampling_top_k_ids,
            sampling_top_k_logits=next_sampling_top_k_logits,
            trace_top_k_ids=next_trace_top_k_ids,
            trace_top_k_logits=next_trace_top_k_logits,
            logsumexp=next_logsumexp,
            output_norm=output_norm,
            layer_outputs=layer_outputs,
        )


class LMState(eqx.Module):
    kv_cache: State
    next_token_position: Int[Array, " batch"]
    root_bonus_id: Int[Array, " batch"]
    root_sample_logits: Float[Array, "batch vocabulary"]
    memory: MemoryBuffers

    @classmethod
    def from_prefill(
        cls,
        prefill_results: "PrefillResults",
        next_token_position: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        root_sample_keys: Key[Array, " batch"],
    ) -> "LMState":
        root_sample_logits = jax.vmap(lambda policy, logits: policy.process_logits(logits.astype(jnp.float32)))(
            sampling_policy,
            prefill_results.last_token_logits,
        )
        root_bonus_id = jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            root_sample_keys,
            root_sample_logits,
        ).astype(jnp.int32)
        return cls(
            kv_cache=prefill_results.state,
            next_token_position=next_token_position,
            root_bonus_id=root_bonus_id,
            root_sample_logits=root_sample_logits,
            memory=prefill_results.memory,
        )

    def create_root_proposal(self, budget: int = 128) -> TrieProposal:
        return TrieProposal.create(
            root_ids=self.root_bonus_id,
            root_sample_positions=self.next_token_position + 1,
            budget=budget,
        )

    def recent_token_ids(self, length: int) -> tuple[Int[Array, "batch length"], Bool[Array, "batch length"]]:
        assert self.memory.token_ids is not None
        return self.memory.token_ids.recent(length)

    def recent_output_norm(
        self,
        length: int,
    ) -> tuple[Float[Array, "batch length channels"], Bool[Array, "batch length"]]:
        assert self.memory.output_norm is not None
        return self.memory.output_norm.recent(length)

    def recent_layer_output(
        self,
        index: int,
        length: int,
    ) -> tuple[Float[Array, "batch length channels"], Bool[Array, "batch length"]]:
        return self.memory.layer_outputs[index].recent(length)

    def commit(
        self,
        state_request: StateRequest,
        decoder_result: DecoderResult,
        processed_tree_logits: Float[Array, "batch nodes vocabulary"],
        accepted: AcceptedProposal,
        emitted_token_ids: Int[Array, "batch max_slots"],
        *,
        sampling_top_k_ids: Int[Array, "batch max_slots top_k"] | None = None,
        sampling_top_k_logits: Float[Array, "batch max_slots top_k"] | None = None,
        trace_top_k_ids: Int[Array, "batch max_slots top_k"] | None = None,
        trace_top_k_logits: Float[Array, "batch max_slots top_k"] | None = None,
        logsumexp: Float[Array, "batch max_slots"] | None = None,
    ) -> "LMState":
        updated_state = decoder_result.updated_state
        assert updated_state is not None, "updated_state should not be None"
        if accepted.compact_indices.shape[1] > 1:
            kv_cache = compact_state_layers(
                updated_state,
                cache_len=self.kv_cache[0].prefix_lengths(),
                accepted_indices=accepted.compact_indices,
                num_accepted=accepted.num_compact_indices,
                max_slots=accepted.compact_indices.shape[1],
            )
        else:
            kv_cache = updated_state

        memory = self.memory.commit(
            state_request,
            decoder_result,
            accepted.compact_indices,
            accepted.num_compact_indices,
            emitted_token_ids,
            sampling_top_k_ids=sampling_top_k_ids,
            sampling_top_k_logits=sampling_top_k_logits,
            trace_top_k_ids=trace_top_k_ids,
            trace_top_k_logits=trace_top_k_logits,
            logsumexp=logsumexp,
        )
        root_sample_logits = processed_tree_logits[
            jnp.arange(accepted.terminal_node_indices.shape[0], dtype=jnp.int32),
            accepted.terminal_node_indices,
        ]
        return LMState(
            kv_cache=kv_cache,
            next_token_position=self.next_token_position + accepted.num_compact_indices,
            root_bonus_id=accepted.bonus_token_ids,
            root_sample_logits=root_sample_logits,
            memory=memory,
        )
