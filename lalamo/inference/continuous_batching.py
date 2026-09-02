import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.models import GenerationConfig, LanguageModel
from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules import Decoder, DecoderForwardPassConfig, State
from lalamo.modules.token_mixer import StateLayerBase
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.kv_cache import KVCacheLayer, PagedKVCacheLayer, StaticKVCacheLayer
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy


@dataclass(frozen=True)
class ContinuousBatchingConfig:
    page_size: int = 32
    total_pages: int = 1_024
    slot_count: int = 64
    prefill_batch_size: int = 32
    prefill_chunk_size: int = 128

    def __post_init__(self) -> None:
        if self.page_size != 32:
            raise ValueError("page_size must be 32.")
        if min(self.total_pages, self.slot_count, self.prefill_batch_size, self.prefill_chunk_size) < 1:
            raise ValueError("Batching sizes must be positive.")


class TokenEvent(NamedTuple):
    token_id: int | None = None
    logprob: float | None = None
    top_token_ids: tuple[int, ...] = ()
    top_logprobs: tuple[float, ...] = ()
    finish_reason: Literal["stop", "length"] | None = None
    completion_tokens: int = 0


class DecodeResult(NamedTuple):
    state: State
    logits: Float[Array, "batch vocabulary"]
    token_ids: Int[Array, "steps batch"]
    token_logprobs: Float[Array, "steps batch"] | None
    top_token_ids: Int[Array, "steps batch top"] | None
    top_logprobs: Float[Array, "steps batch top"] | None
    sampling_keys: Key[Array, " batch"]
    sampling_policy: SamplingPolicy
    invalid_logits: Bool[Array, ""]


@eqx.filter_jit(donate="all-except-first")
def _decode(
    decoder: Decoder,
    state: State,
    logits: Float[Array, "batch vocabulary"],
    token_positions: Int[Array, "steps batch 1"],
    sampling_policy: SamplingPolicy,
    sampling_keys: Key[Array, " batch"],
    return_logprobs: bool,
) -> DecodeResult:
    def step(
        carry: tuple[State, Array, SamplingPolicy, Array, Array], positions: Array
    ) -> tuple[
        tuple[State, Array, SamplingPolicy, Array, Array],
        tuple[Array, Array | None, Array | None, Array | None],
    ]:
        state, logits, sampling_policy, sampling_keys, invalid_logits = carry
        invalid_logits |= ~jnp.isfinite(logits).all()
        if sampling_policy.is_greedy:
            sample_keys = sampling_keys
            next_keys = sampling_keys
        else:
            split_keys = jax.vmap(jax.random.split)(sampling_keys)
            next_keys, sample_keys = split_keys[:, 0], split_keys[:, 1]

        token_ids = call_vmapped(
            lambda policy, row, key: policy(row, keychain=Keychain(key, key, decoder.sharding_config)),
            sampling_policy,
            logits.astype(jnp.float32),
            sample_keys,
        )
        if return_logprobs:
            normalized = jax.nn.log_softmax(logits.astype(jnp.float32))
            normalized = jnp.where(jnp.isneginf(normalized), -9999.0, normalized)
            token_logprobs = jnp.take_along_axis(normalized, token_ids[:, None], axis=1)[:, 0]
            top_values, top_ids = jax.lax.top_k(normalized, 20)
        else:
            token_logprobs = None
            top_values = None
            top_ids = None

        if sampling_policy.has_count_penalties:
            sampling_policy = call_vmapped(
                lambda policy, token_id: policy.with_next_token_count(token_id), sampling_policy, token_ids
            )
        decoded = decoder(
            token_ids[:, None],
            positions,
            state=state,
            return_updated_state=True,
            forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
            keychain=Keychain.init(0, sharding_config=decoder.sharding_config),
        )
        assert decoded.updated_state is not None
        carry = (
            decoded.updated_state,
            decoded.logits[:, 0].astype(jnp.float32),
            sampling_policy,
            next_keys,
            invalid_logits,
        )
        return carry, (token_ids, token_logprobs, top_ids, top_values)

    carry, generated = jax.lax.scan(
        step, (state, logits, sampling_policy, sampling_keys, jnp.zeros((), dtype=jnp.bool_)), token_positions
    )
    state, logits, sampling_policy, sampling_keys, invalid_logits = carry
    token_ids, token_logprobs, top_ids, top_values = generated
    return DecodeResult(
        state, logits, token_ids, token_logprobs, top_ids, top_values, sampling_keys, sampling_policy, invalid_logits
    )


@eqx.filter_jit(donate="all")
def _merge_prefill(
    state: State,
    logits: Float[Array, "slots vocabulary"],
    prefilled_state: State,
    prefilled_logits: Float[Array, "batch vocabulary"],
    tables: Int[Array, "batch pages"],
    slots: Int[Array, " batch"],
    prefill_pages: int,
    page_size: int,
) -> tuple[State, Float[Array, "slots vocabulary"]]:
    layers: list[StateLayerBase] = []
    for pool, prefill in zip(state, prefilled_state, strict=True):
        if isinstance(pool, PagedKVCacheLayer):
            assert isinstance(prefill, StaticKVCacheLayer)
            keys, values = (
                rearrange(
                    cache[:, : prefill_pages * page_size],
                    "batch (pages page) groups channels -> groups batch pages page channels",
                    pages=prefill_pages,
                    page=page_size,
                )
                for cache in (prefill.keys, prefill.values)
            )
            layers.append(
                PagedKVCacheLayer(
                    pool.keys.at[:, tables].set(keys),
                    pool.values.at[:, tables].set(values),
                    pool.block_tables,
                    pool.lengths,
                )
            )
        else:
            assert not isinstance(prefill, KVCacheLayer)
            layers.append(jax.tree.map(lambda old, new: old.at[slots].set(new), pool, prefill))
    return State(layers), logits.at[slots].set(prefilled_logits)


@dataclass(eq=False)
class BatchingSequence:
    prompt_token_ids: tuple[int, ...]
    max_output_length: int
    stop_token_ids: tuple[int, ...]
    events: SimpleQueue[TokenEvent]
    wake: Callable[[], object] | None
    sampling_policy: SamplingPolicy
    sampling_key: Key[Array, ""]
    return_logprobs: bool
    output_token_ids: list[int]
    pages: tuple[int, ...] = ()
    slot: int | None = None


class ContinuousBatchingEngine:
    def __init__(self, model: LanguageModel, config: ContinuousBatchingConfig) -> None:
        if model.sharding_config.resolve_axis(LogicalAxis.BATCH) is not None:
            raise ValueError("Continuous batching does not support batch-sharded models.")
        self.model = model
        self.config = config
        self.context_limit = min(
            config.page_size * config.total_pages,
            *(rope.config.max_sequence_length for rope in model.decoder.transformer.ropes),
        )

        state_dtype = DecoderForwardPassConfig.for_inference().embedding_forward_pass_config.activation_dtype
        static_state = model.decoder.init_static_state(config.slot_count, 1, state_dtype)
        layers: list[StateLayerBase] = []
        for owner, static_layer in zip(model.decoder.transformer.kv_source_layer_indices, static_state, strict=True):
            mixer = model.decoder.transformer.layers[owner].mixer
            if isinstance(mixer, Attention):
                if (
                    model.sharding_config.mesh.devices.flat[0].platform != "gpu"
                    or mixer.has_sinks
                    or (mixer.config.sliding_window_size is not None and mixer.config.logit_soft_cap is not None)
                ):
                    raise ValueError("Paged batching requires GPU attention without sinks or soft-capped windows.")
                shape = (mixer.config.num_groups, config.total_pages + 1, config.page_size, mixer.config.head_dim)
                layers.append(
                    PagedKVCacheLayer(
                        jnp.zeros(shape, dtype=state_dtype),
                        jnp.zeros(shape, dtype=state_dtype),
                        jnp.empty((0, 0), dtype=jnp.int32),
                        jnp.empty((0,), dtype=jnp.int32),
                    )
                )
            else:
                layers.append(static_layer)
        self._state = State(layers)
        self._last_logits = jnp.zeros((config.slot_count, model.decoder.vocab_size), dtype=jnp.float32)
        self._incoming: SimpleQueue[BatchingSequence] = SimpleQueue()
        self._pending: deque[BatchingSequence] = deque()
        self._active: deque[BatchingSequence] = deque()
        self._free_slots: deque[int] = deque(range(config.slot_count))
        self._free_pages: deque[int] = deque(range(config.total_pages))

    def submit(
        self,
        prompt_token_ids: tuple[int, ...],
        max_output_length: int,
        generation_config: GenerationConfig,
        seed: int,
        *,
        return_logprobs: bool = False,
        wake: Callable[[], object] | None = None,
    ) -> SimpleQueue[TokenEvent]:
        if not prompt_token_ids or max_output_length < 1:
            raise ValueError("A non-empty prompt and positive max_output_length are required.")
        if seed < 0 or seed >= 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1.")
        if any(token_id < 0 or token_id >= self.model.decoder.vocab_size for token_id in prompt_token_ids):
            raise ValueError("prompt_token_ids contain an out-of-vocabulary token.")
        if len(prompt_token_ids) + max_output_length > self.context_limit:
            raise ValueError("The prompt and requested output exceed the configured context capacity.")
        sampling_policy = generation_config.default_policy()
        if sampling_policy.has_count_penalties:
            prompt = jnp.asarray(prompt_token_ids, dtype=jnp.int32)
            sampling_policy = sampling_policy.with_prompt_token_counts(
                prompt, jnp.asarray(len(prompt)), self.model.decoder.vocab_size
            )

        token_events: SimpleQueue[TokenEvent] = SimpleQueue()
        self._incoming.put(
            BatchingSequence(
                prompt_token_ids=prompt_token_ids,
                max_output_length=max_output_length,
                stop_token_ids=generation_config.stop_token_ids,
                events=token_events,
                wake=wake,
                sampling_policy=sampling_policy,
                sampling_key=jax.random.key(seed),
                return_logprobs=return_logprobs,
                output_token_ids=[],
            )
        )
        return token_events

    def step(self) -> bool:
        while not self._incoming.empty():
            self._pending.append(self._incoming.get())

        if self._pending and self._free_slots:
            prefill_sequences: list[BatchingSequence] = []
            pages_needed = 0
            for sequence in tuple(self._pending):
                sequence_length = len(sequence.prompt_token_ids) + len(sequence.output_token_ids)
                page_count = math.ceil((sequence_length + 1) / self.config.page_size)
                if pages_needed + page_count > len(self._free_pages):
                    break
                prefill_sequences.append(sequence)
                pages_needed += page_count
                if len(prefill_sequences) == min(self.config.prefill_batch_size, len(self._free_slots)):
                    break

            if prefill_sequences:
                token_capacity = (
                    1
                    << (
                        max(
                            len(sequence.prompt_token_ids) + len(sequence.output_token_ids)
                            for sequence in prefill_sequences
                        )
                        - 1
                    ).bit_length()
                )
                batch_size = 1 << (len(prefill_sequences) - 1).bit_length()
                padded_sequences = [
                    *prefill_sequences,
                    *([prefill_sequences[0]] * (batch_size - len(prefill_sequences))),
                ]
                slot_ids = tuple(self._free_slots.popleft() for _ in prefill_sequences)
                for sequence, slot in zip(prefill_sequences, slot_ids, strict=True):
                    sequence_length = len(sequence.prompt_token_ids) + len(sequence.output_token_ids)
                    page_count = math.ceil((sequence_length + 1) / self.config.page_size)
                    sequence.pages = tuple(self._free_pages.popleft() for _ in range(page_count))
                    sequence.slot = slot

                padded_slots = (*slot_ids, *((slot_ids[0],) * (batch_size - len(prefill_sequences))))
                prefill_pages = math.ceil(token_capacity / self.config.page_size)
                tables_array = jnp.asarray(
                    [
                        (
                            *sequence.pages[:prefill_pages],
                            *((self.config.total_pages,) * (prefill_pages - len(sequence.pages))),
                        )
                        for sequence in padded_sequences
                    ],
                    dtype=jnp.int32,
                )
                tokens = np.zeros((batch_size, token_capacity), dtype=np.int32)
                for row, sequence in enumerate(padded_sequences):
                    context = (*sequence.prompt_token_ids, *sequence.output_token_ids)
                    tokens[row, : len(context)] = context
                lengths = jnp.asarray(
                    [len(sequence.prompt_token_ids) + len(sequence.output_token_ids) for sequence in padded_sequences],
                    dtype=jnp.int32,
                )
                prefilled = self.model.prefill_tokens(
                    jnp.asarray(tokens),
                    prefill_pages * self.config.page_size,
                    lengths,
                    chunk_size=self.config.prefill_chunk_size,
                    keychain=Keychain.init(0, sharding_config=self.model.sharding_config),
                )
                slots_array = jnp.asarray(padded_slots, dtype=jnp.int32)
                self._state, self._last_logits = _merge_prefill(
                    self._state,
                    self._last_logits,
                    prefilled.state,
                    prefilled.last_token_logits,
                    tables_array,
                    slots_array,
                    prefill_pages,
                    self.config.page_size,
                )
                for sequence in prefill_sequences:
                    self._pending.remove(sequence)
                    self._active.append(sequence)
                return True

        if not self._active:
            return False

        while True:
            first_sequence = self._active[0]
            compatible_sequences = [
                sequence
                for sequence in self._active
                if jax.tree.structure(sequence.sampling_policy) == jax.tree.structure(first_sequence.sampling_policy)
                and sequence.return_logprobs == first_sequence.return_logprobs
            ]
            available_block_size = min(
                64,
                *(sequence.max_output_length - len(sequence.output_token_ids) for sequence in compatible_sequences),
            )
            available_token_capacities = [
                (len(sequence.pages) + len(self._free_pages)) * self.config.page_size
                - len(sequence.prompt_token_ids)
                - len(sequence.output_token_ids)
                for sequence in compatible_sequences
            ]
            available_block_size = min(available_block_size, max(available_token_capacities))
            if available_block_size:
                block_size = 1 << (available_block_size.bit_length() - 1)
                decode_sequences = []
                for sequence in compatible_sequences:
                    sequence_length = len(sequence.prompt_token_ids) + len(sequence.output_token_ids)
                    pages_needed = math.ceil((sequence_length + block_size) / self.config.page_size) - len(
                        sequence.pages
                    )
                    if pages_needed > len(self._free_pages):
                        continue
                    sequence.pages += tuple(self._free_pages.popleft() for _ in range(pages_needed))
                    decode_sequences.append(sequence)
                break

            if len(self._active) == 1:
                raise RuntimeError("The sole active sequence cannot grow despite passing capacity validation.")

            preempted_sequence = max(self._active, key=lambda sequence: len(sequence.pages))
            assert preempted_sequence.slot is not None
            self._active.remove(preempted_sequence)
            self._free_slots.append(preempted_sequence.slot)
            self._free_pages.extend(preempted_sequence.pages)
            preempted_sequence.slot = None
            preempted_sequence.pages = ()
            self._pending.appendleft(preempted_sequence)

        batch_size = 1 << (len(decode_sequences) - 1).bit_length()
        padded_sequences = [
            *decode_sequences,
            *([decode_sequences[0]] * (batch_size - len(decode_sequences))),
        ]
        pages_per_sequence = 1 << (max(len(sequence.pages) for sequence in padded_sequences) - 1).bit_length()
        block_tables = jnp.asarray(
            [
                (*sequence.pages, *((self.config.total_pages,) * (pages_per_sequence - len(sequence.pages))))
                for sequence in padded_sequences
            ],
            dtype=jnp.int32,
        )
        slot_indices = jnp.asarray([sequence.slot for sequence in padded_sequences], dtype=jnp.int32)
        sequence_lengths = jnp.asarray(
            [len(sequence.prompt_token_ids) + len(sequence.output_token_ids) for sequence in padded_sequences],
            dtype=jnp.int32,
        )
        sampling_policies = jax.tree.map(
            lambda *leaves: jnp.stack(leaves),
            *(sequence.sampling_policy for sequence in padded_sequences),
        )
        sampling_keys = jnp.stack([sequence.sampling_key for sequence in padded_sequences])

        decode_layers: list[StateLayerBase] = []
        for layer in self._state:
            if isinstance(layer, PagedKVCacheLayer):
                decode_layers.append(
                    PagedKVCacheLayer(
                        layer.keys,
                        layer.values,
                        block_tables.copy(),
                        sequence_lengths.copy(),
                    )
                )
            else:
                decode_layers.append(jax.tree.map(lambda array: array[slot_indices], layer))

        token_positions = sequence_lengths[None, :, None] + jnp.arange(block_size, dtype=jnp.int32)[:, None, None]
        decoded = _decode(
            self.model.decoder,
            State(decode_layers),
            self._last_logits[slot_indices],
            token_positions,
            sampling_policies,
            sampling_keys,
            first_sequence.return_logprobs,
        )

        updated_layers = []
        for pool, update in zip(self._state, decoded.state, strict=True):
            if isinstance(pool, PagedKVCacheLayer):
                assert isinstance(update, PagedKVCacheLayer)
                updated_layers.append(PagedKVCacheLayer(update.keys, update.values, pool.block_tables, pool.lengths))
            else:
                updated_layers.append(jax.tree.map(lambda old, new: old.at[slot_indices].set(new), pool, update))
        self._state = State(updated_layers)
        self._last_logits = self._last_logits.at[slot_indices].set(decoded.logits)

        token_ids = np.asarray(decoded.token_ids)
        token_logprobs = None if decoded.token_logprobs is None else np.asarray(decoded.token_logprobs)
        top_token_ids = None if decoded.top_token_ids is None else np.asarray(decoded.top_token_ids)
        top_logprobs = None if decoded.top_logprobs is None else np.asarray(decoded.top_logprobs)
        if bool(np.asarray(decoded.invalid_logits)):
            raise FloatingPointError("Model produced non-finite logits.")

        for row, sequence in enumerate(decode_sequences):
            reached_stop_token = False
            for step_index in range(block_size):
                token_id = int(token_ids[step_index, row])
                sequence.output_token_ids.append(token_id)
                reached_stop_token = token_id in sequence.stop_token_ids
                if not reached_stop_token:
                    if token_logprobs is None:
                        sequence.events.put(TokenEvent(token_id))
                    else:
                        assert top_token_ids is not None and top_logprobs is not None
                        sequence.events.put(
                            TokenEvent(
                                token_id,
                                float(token_logprobs[step_index, row]),
                                tuple(int(value) for value in top_token_ids[step_index, row]),
                                tuple(float(value) for value in top_logprobs[step_index, row]),
                            )
                        )
                if reached_stop_token or len(sequence.output_token_ids) == sequence.max_output_length:
                    break

            self._active.remove(sequence)
            if reached_stop_token or len(sequence.output_token_ids) == sequence.max_output_length:
                assert sequence.slot is not None
                self._free_slots.append(sequence.slot)
                self._free_pages.extend(sequence.pages)
                sequence.events.put(
                    TokenEvent(
                        finish_reason="stop" if reached_stop_token else "length",
                        completion_tokens=len(sequence.output_token_ids),
                    )
                )
            else:
                sequence.sampling_key = decoded.sampling_keys[row]
                sequence.sampling_policy = jax.tree.map(lambda leaf, row=row: leaf[row], decoded.sampling_policy)
                self._active.append(sequence)

            if sequence.wake is not None:
                sequence.wake()

        return True
