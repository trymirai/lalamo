import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from queue import SimpleQueue
from typing import ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.models import GenerationConfig, LanguageModel
from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules import Decoder, DecoderForwardPassConfig, State
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.kv_cache import PagedKVCacheLayer, PagedKVCachePool, StaticKVCacheLayer
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy


@dataclass(frozen=True)
class ContinuousBatchingConfig:
    total_pages: int = 1_024
    slot_count: int = 64
    prefill_batch_size: int = 32
    prefill_chunk_size: int = 128
    page_size: ClassVar[int] = 32


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"


class TokenLogprobs(NamedTuple):
    logprob: float
    top_token_ids: tuple[int, ...]
    top_logprobs: tuple[float, ...]


class GeneratedToken(NamedTuple):
    token_id: int
    logprobs: TokenLogprobs | None


class SequenceFinished(NamedTuple):
    reason: FinishReason
    completion_tokens: int


type TokenEvent = GeneratedToken | SequenceFinished
type _DecodeCarry = tuple[State, Array, SamplingPolicy, Array, Array]


class DecodedLogprobs(NamedTuple):
    token_logprobs: Float[Array, "steps batch"]
    top_token_ids: Int[Array, "steps batch top"]
    top_logprobs: Float[Array, "steps batch top"]


class DecodeResult(NamedTuple):
    state: State
    logits: Float[Array, "batch vocabulary"]
    token_ids: Int[Array, "steps batch"]
    logprobs: DecodedLogprobs | None
    sampling_keys: Key[Array, " batch"]
    sampling_policy: SamplingPolicy
    invalid_logits: Bool[Array, ""]


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _pad_to_power_of_two[T](batch: list[T]) -> list[T]:
    return batch + [batch[0]] * (_next_power_of_two(len(batch)) - len(batch))


@eqx.filter_jit(donate="all-except-first")
def _decode(
    decoder: Decoder,
    state: State,
    block_tables: Int[Array, "batch pages_per_sequence"],
    lengths: Int[Array, " batch"],
    logits: Float[Array, "batch vocabulary"],
    token_positions: Int[Array, "steps batch 1"],
    sampling_policy: SamplingPolicy,
    sampling_keys: Key[Array, " batch"],
    return_logprobs: bool,
) -> DecodeResult:
    def step(carry: _DecodeCarry, positions: Array) -> tuple[_DecodeCarry, tuple[Array, DecodedLogprobs | None]]:
        state, logits, sampling_policy, sampling_keys, invalid_logits = carry
        invalid_logits |= ~jnp.isfinite(logits).all()
        if sampling_policy.is_greedy:
            sample_keys = next_keys = sampling_keys
        else:
            next_keys, sample_keys = jnp.unstack(jax.vmap(jax.random.split)(sampling_keys), axis=1)

        token_ids = call_vmapped(
            lambda policy, row, key: policy(row, keychain=Keychain(key, key, decoder.sharding_config)),
            sampling_policy,
            logits,
            sample_keys,
        )
        logprobs = None
        if return_logprobs:
            normalized = jax.nn.log_softmax(logits)
            normalized = jnp.where(jnp.isneginf(normalized), -9999.0, normalized)
            top_logprobs, top_token_ids = jax.lax.top_k(normalized, 20)
            token_logprobs = jnp.take_along_axis(normalized, token_ids[:, None], axis=1)[:, 0]
            logprobs = DecodedLogprobs(token_logprobs, top_token_ids, top_logprobs)
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
        return carry, (token_ids, logprobs)

    views = State(
        PagedKVCacheLayer(layer.keys, layer.values, block_tables, lengths)
        if isinstance(layer, PagedKVCachePool)
        else layer
        for layer in state
    )
    initial_carry = (views, logits, sampling_policy, sampling_keys, jnp.zeros((), dtype=jnp.bool_))
    (views, logits, sampling_policy, sampling_keys, invalid_logits), (token_ids, logprobs) = jax.lax.scan(
        step, initial_carry, token_positions
    )
    pools = State(
        PagedKVCachePool(layer.keys, layer.values) if isinstance(layer, PagedKVCacheLayer) else layer
        for layer in views
    )
    return DecodeResult(pools, logits, token_ids, logprobs, sampling_keys, sampling_policy, invalid_logits)


@eqx.filter_jit(donate="all")
def _merge_prefill(
    state: State,
    logits: Float[Array, "slots vocabulary"],
    prefilled_state: State,
    prefilled_logits: Float[Array, "batch vocabulary"],
    page_indices: Int[Array, "batch pages"],
    slots: Int[Array, " batch"],
) -> tuple[State, Float[Array, "slots vocabulary"]]:
    layers = []
    for pool, prefill in zip(state, prefilled_state, strict=True):
        if isinstance(pool, PagedKVCachePool):
            assert isinstance(prefill, StaticKVCacheLayer)
            tokens = page_indices.shape[1] * pool.page_size
            layers.append(pool.write_pages(page_indices, prefill.keys[:, :tokens], prefill.values[:, :tokens]))
        else:
            layers.append(jax.tree.map(lambda old, new: old.at[slots].set(new), pool, prefill))
    return State(layers), logits.at[slots].set(prefilled_logits)


@dataclass(eq=False)
class BatchingSequence:
    prompt_token_ids: tuple[int, ...]
    max_output_length: int
    stop_token_ids: tuple[int, ...]
    on_events: Callable[[Sequence[TokenEvent]], object]
    sampling_policy: SamplingPolicy
    sampling_key: Key[Array, ""]
    return_logprobs: bool
    output_token_ids: list[int] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    slot: int | None = None

    @property
    def length(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def remaining_output_length(self) -> int:
        return self.max_output_length - len(self.output_token_ids)


class ContinuousBatchingEngine:
    def __init__(self, model: LanguageModel, config: ContinuousBatchingConfig) -> None:
        if model.sharding_config.resolve_axis(LogicalAxis.BATCH) is not None:
            raise ValueError("Continuous batching does not support batch-sharded models.")
        self.model = model
        self.config = config
        transformer = model.decoder.transformer
        self.context_limit = min(
            config.page_size * config.total_pages,
            *(rope.config.max_sequence_length for rope in transformer.ropes),
        )

        state_dtype = DecoderForwardPassConfig.for_inference().embedding_forward_pass_config.activation_dtype
        static_state = model.decoder.init_static_state(config.slot_count, 1, state_dtype)
        layers = []
        for owner_index, static_layer in zip(transformer.kv_source_layer_indices, static_state, strict=True):
            mixer = transformer.layers[owner_index].mixer
            if not isinstance(mixer, Attention):
                layers.append(static_layer)
                continue
            if (
                model.sharding_config.mesh.devices.flat[0].platform != "gpu"
                or mixer.has_sinks
                or (mixer.config.sliding_window_size is not None and mixer.config.logit_soft_cap is not None)
            ):
                raise ValueError("Paged batching requires GPU attention without sinks or soft-capped windows.")
            # The extra page at index total_pages absorbs writes from padded batch rows and unallocated table entries.
            shape = (mixer.config.num_groups, config.total_pages + 1, config.page_size, mixer.config.head_dim)
            layers.append(PagedKVCachePool(jnp.zeros(shape, dtype=state_dtype), jnp.zeros(shape, dtype=state_dtype)))
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
        on_events: Callable[[Sequence[TokenEvent]], object],
    ) -> None:
        if len(prompt_token_ids) + max_output_length > self.context_limit:
            raise ValueError("The prompt and requested output exceed the configured context capacity.")
        sampling_policy = generation_config.default_policy()
        if sampling_policy.has_count_penalties:
            prompt = jnp.asarray(prompt_token_ids, dtype=jnp.int32)
            sampling_policy = sampling_policy.with_prompt_token_counts(
                prompt, jnp.asarray(len(prompt)), self.model.decoder.vocab_size
            )
        self._incoming.put(
            BatchingSequence(
                prompt_token_ids=prompt_token_ids,
                max_output_length=max_output_length,
                stop_token_ids=generation_config.stop_token_ids,
                on_events=on_events,
                sampling_policy=sampling_policy,
                sampling_key=jax.random.key(seed),
                return_logprobs=return_logprobs,
            )
        )

    def step(self) -> bool:
        while not self._incoming.empty():
            self._pending.append(self._incoming.get())
        if self._prefill():
            return True
        if not self._active:
            return False
        self._decode()
        return True

    def _reserve_pages(self, sequence: BatchingSequence, new_tokens: int) -> bool:
        page_count = math.ceil((sequence.length + new_tokens) / self.config.page_size) - len(sequence.pages)
        if page_count > len(self._free_pages):
            return False
        sequence.pages.extend(self._free_pages.popleft() for _ in range(page_count))
        return True

    def _release(self, sequence: BatchingSequence) -> None:
        assert sequence.slot is not None
        self._free_slots.append(sequence.slot)
        self._free_pages.extend(sequence.pages)
        sequence.slot = None
        sequence.pages = []

    def _block_tables(self, batch: list[BatchingSequence], pages_per_sequence: int) -> Int[Array, "batch pages"]:
        padding = [self.config.total_pages] * pages_per_sequence
        return jnp.asarray([(sequence.pages + padding)[:pages_per_sequence] for sequence in batch], dtype=jnp.int32)

    def _prefill(self) -> bool:
        batch: list[BatchingSequence] = []
        while (
            self._pending
            and self._free_slots
            and len(batch) < self.config.prefill_batch_size
            and self._reserve_pages(self._pending[0], 1)
        ):
            sequence = self._pending.popleft()
            sequence.slot = self._free_slots.popleft()
            batch.append(sequence)
        if not batch:
            return False
        self._active.extend(batch)

        token_capacity = _next_power_of_two(max(sequence.length for sequence in batch))
        pages_per_sequence = math.ceil(token_capacity / self.config.page_size)
        padded = _pad_to_power_of_two(batch)
        tokens = np.zeros((len(padded), token_capacity), dtype=np.int32)
        for row, sequence in enumerate(padded):
            tokens[row, : sequence.length] = (*sequence.prompt_token_ids, *sequence.output_token_ids)
        prefilled = self.model.prefill_tokens(
            jnp.asarray(tokens),
            pages_per_sequence * self.config.page_size,
            jnp.asarray([sequence.length for sequence in padded], dtype=jnp.int32),
            chunk_size=self.config.prefill_chunk_size,
            keychain=Keychain.init(0, sharding_config=self.model.sharding_config),
        )
        self._state, self._last_logits = _merge_prefill(
            self._state,
            self._last_logits,
            prefilled.state,
            prefilled.last_token_logits,
            self._block_tables(padded, pages_per_sequence),
            jnp.asarray([sequence.slot for sequence in padded], dtype=jnp.int32),
        )
        return True

    def _plan_decode(self) -> tuple[list[BatchingSequence], int] | None:
        first, *_ = self._active
        batch = [
            sequence
            for sequence in self._active
            if sequence.return_logprobs == first.return_logprobs
            and jax.tree.structure(sequence.sampling_policy) == jax.tree.structure(first.sampling_policy)
        ]
        growable_tokens = max(
            (len(sequence.pages) + len(self._free_pages)) * self.config.page_size - sequence.length
            for sequence in batch
        )
        block_size = min(64, growable_tokens, *(sequence.remaining_output_length for sequence in batch))
        if block_size < 1:
            return None
        block_size = 1 << (block_size.bit_length() - 1)
        return [sequence for sequence in batch if self._reserve_pages(sequence, block_size)], block_size

    def _preempt(self) -> None:
        if len(self._active) == 1:
            raise RuntimeError("The sole active sequence cannot grow despite passing capacity validation.")
        victim = max(self._active, key=lambda sequence: len(sequence.pages))
        self._active.remove(victim)
        self._release(victim)
        self._pending.appendleft(victim)

    def _decode(self) -> None:
        while (plan := self._plan_decode()) is None:
            self._preempt()
        batch, block_size = plan
        padded = _pad_to_power_of_two(batch)
        slots = jnp.asarray([sequence.slot for sequence in padded], dtype=jnp.int32)
        lengths = jnp.asarray([sequence.length for sequence in padded], dtype=jnp.int32)
        block_tables = self._block_tables(padded, _next_power_of_two(max(len(sequence.pages) for sequence in padded)))
        token_positions = lengths[None, :, None] + jnp.arange(block_size, dtype=jnp.int32)[:, None, None]
        rows = State(
            layer if isinstance(layer, PagedKVCachePool) else jax.tree.map(lambda array: array[slots], layer)
            for layer in self._state
        )
        decoded = _decode(
            self.model.decoder,
            rows,
            block_tables,
            lengths,
            self._last_logits[slots],
            token_positions,
            jax.tree.map(lambda *leaves: jnp.stack(leaves), *(sequence.sampling_policy for sequence in padded)),
            jnp.stack([sequence.sampling_key for sequence in padded]),
            batch[0].return_logprobs,
        )
        self._state = State(
            new
            if isinstance(new, PagedKVCachePool)
            else jax.tree.map(lambda old, new: old.at[slots].set(new), old, new)
            for old, new in zip(self._state, decoded.state, strict=True)
        )
        self._last_logits = self._last_logits.at[slots].set(decoded.logits)
        if bool(decoded.invalid_logits):
            raise FloatingPointError("Model produced non-finite logits.")

        token_ids = np.asarray(decoded.token_ids)
        logprobs = None if decoded.logprobs is None else jax.tree.map(np.asarray, decoded.logprobs)
        for row, sequence in enumerate(batch):
            self._active.remove(sequence)
            sequence.sampling_key = decoded.sampling_keys[row]
            sequence.sampling_policy = jax.tree.map(lambda leaf, row=row: leaf[row], decoded.sampling_policy)
            events: list[TokenEvent] = []
            for step, token_id in enumerate(map(int, token_ids[:, row])):
                sequence.output_token_ids.append(token_id)
                if token_id in sequence.stop_token_ids:
                    events.append(SequenceFinished(FinishReason.STOP, len(sequence.output_token_ids)))
                    break
                token_logprobs = None
                if logprobs is not None:
                    token_logprobs = TokenLogprobs(
                        float(logprobs.token_logprobs[step, row]),
                        tuple(map(int, logprobs.top_token_ids[step, row])),
                        tuple(map(float, logprobs.top_logprobs[step, row])),
                    )
                events.append(GeneratedToken(token_id, token_logprobs))
                if sequence.remaining_output_length == 0:
                    events.append(SequenceFinished(FinishReason.LENGTH, len(sequence.output_token_ids)))
                    break
            if isinstance(events[-1], SequenceFinished):
                self._release(sequence)
            else:
                self._active.append(sequence)
            sequence.on_events(events)
