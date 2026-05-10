from dataclasses import dataclass, field
from typing import Annotated

from annotated_types import Ge
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from lalamo.modules.decoder import DecoderActivationTrace, DecoderResult
from lalamo.modules.token_mixers.state.common import State

__all__ = [
    "LMState",
    "MemoryBuffers",
    "RingBuffer",
    "StateRequest",
]


@dataclass(frozen=True)
class StateRequest:
    token_id_capacity: Annotated[int, Ge(0)] = 0
    output_norm_capacity: Annotated[int, Ge(0)] = 0
    layer_capacity: Annotated[int, Ge(0)] = 0
    layer_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        assert (
            self.layer_capacity > 0
        ) == (self.layer_indices != ()), "layer_capacity and layer_indices must both be set or both be empty."


@dataclass(frozen=True)
class RingBuffer:
    values: Array
    length: Int[Array, ""] = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

    @classmethod
    def from_rows(cls, rows: Array, capacity: int) -> "RingBuffer":
        return cls(jnp.zeros((capacity, *rows.shape[1:]), dtype=rows.dtype)).append(
            rows,
            rows.shape[0],
        )

    def append(self, rows: Array, count: Int[Array, ""] | int) -> "RingBuffer":
        assert count <= rows.shape[0], "count must not exceed rows.shape[0]."
        count = jnp.asarray(count, dtype=jnp.int32)
        capacity = self.values.shape[0]
        slots = jnp.arange(capacity, dtype=jnp.int32)
        write_count = jnp.minimum(count, capacity)
        write_slots = slots < write_count
        padded_rows = jnp.concatenate([jnp.zeros((1, *rows.shape[1:]), dtype=rows.dtype), rows], axis=0)
        source_indices = jnp.where(write_slots, count - write_count + slots + 1, 0)
        positions = (self.length + count - write_count + slots) % capacity
        retained_rows = padded_rows[source_indices]
        current_rows = self.values[positions]
        write_mask = write_slots.reshape((-1, *([1] * (rows.ndim - 1))))
        values = self.values.at[positions].set(jnp.where(write_mask, retained_rows, current_rows))
        return RingBuffer(values, self.length + count)

    def recent(self, length: int) -> tuple[Array, Bool[Array, " length"]]:
        retained_length = jnp.minimum(
            jnp.minimum(jnp.asarray(length, dtype=jnp.int32), self.length),
            self.values.shape[0],
        )
        start = self.length - retained_length
        slots = jnp.arange(length, dtype=jnp.int32)
        positions = (slots + start) % self.values.shape[0]
        return self.values[positions], slots < retained_length


@dataclass(frozen=True)
class MemoryBuffers:
    token_ids: RingBuffer | None = None
    output_norm: RingBuffer | None = None
    layer_outputs: tuple[RingBuffer, ...] = ()

    @classmethod
    def from_prefill(cls, state_request: StateRequest, activation_trace: DecoderActivationTrace) -> "MemoryBuffers":
        token_ids = None
        if state_request.token_id_capacity > 0:
            token_ids = RingBuffer.from_rows(activation_trace.token_ids[0], state_request.token_id_capacity)

        output_norm = None
        if state_request.output_norm_capacity > 0:
            output_norm = RingBuffer.from_rows(activation_trace.output_norm[0], state_request.output_norm_capacity)

        layer_outputs = tuple(
            RingBuffer.from_rows(
                activation_trace.layer_results[layer_index].outputs[0],
                state_request.layer_capacity,
            )
            for layer_index in state_request.layer_indices
        )

        return MemoryBuffers(
            token_ids=token_ids,
            output_norm=output_norm,
            layer_outputs=layer_outputs,
        )

    def commit(
        self,
        state_request: StateRequest,
        decoder_result: DecoderResult,
        compact_indices: Int[Array, " max_slots"],
        num_compact_indices: int,
    ) -> "MemoryBuffers":
        activation_trace = decoder_result.activation_trace
        assert activation_trace is not None

        def compact_rows(values: Array) -> Array:
            slots = jnp.arange(compact_indices.shape[0], dtype=jnp.int32)
            row_indices = jnp.where(slots < num_compact_indices, compact_indices, 0)
            return values[row_indices]

        token_ids = self.token_ids
        if token_ids is not None:
            token_ids = token_ids.append(compact_rows(activation_trace.token_ids[0]), num_compact_indices)

        output_norm = self.output_norm
        if output_norm is not None:
            output_norm = output_norm.append(compact_rows(activation_trace.output_norm[0]), num_compact_indices)

        layer_outputs = tuple(
            previous_layer_output.append(
                compact_rows(activation_trace.layer_results[layer_index].outputs[0]),
                num_compact_indices,
            )
            for previous_layer_output, layer_index in zip(
                self.layer_outputs,
                state_request.layer_indices,
                strict=True,
            )
        )

        return MemoryBuffers(
            token_ids=token_ids,
            output_norm=output_norm,
            layer_outputs=layer_outputs,
        )


@dataclass(frozen=True)
class LMState:
    kv_cache: State
    next_token_position: Int[Array, ""]
    root_bonus_id: Int[Array, ""]
    memory: MemoryBuffers = field(default_factory=MemoryBuffers)

    def recent_token_ids(self, length: int) -> tuple[Int[Array, " length"], Bool[Array, " length"]]:
        assert self.memory.token_ids is not None
        return self.memory.token_ids.recent(length)

    def recent_output_norm(self, length: int) -> tuple[Float[Array, "length channels"], Bool[Array, " length"]]:
        assert self.memory.output_norm is not None
        return self.memory.output_norm.recent(length)

    def recent_layer_output(
        self,
        index: int,
        length: int,
    ) -> tuple[Float[Array, "length channels"], Bool[Array, " length"]]:
        return self.memory.layer_outputs[index].recent(length)
