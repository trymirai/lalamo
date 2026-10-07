import logging
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from queue import SimpleQueue
from threading import Event
from typing import ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, Key

from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.language_model import PrefillResults
from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules import Decoder, DecoderForwardPassConfig, State
from lalamo.modules.token_mixer import StateLayerBase
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.kv_cache import PagedKVCacheLayer, PagedKVCachePool, StaticKVCacheLayer
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy

__all__ = [
    "ContinuousBatchingConfig",
    "ContinuousBatchingEngine",
    "FinishReason",
    "GeneratedToken",
    "SequenceFinished",
    "TokenEvent",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContinuousBatchingConfig:
    """A `total_pages` of None sizes the KV pool from the device memory left after the model is loaded; a
    `slot_count` of None admits as many concurrent sequences as can each reach the full context length."""

    total_pages: int | None = None
    slot_count: int | None = None
    max_context_length: int | None = None
    prefill_batch_size: int = 1
    prefill_chunk_size: int = 512
    page_size: ClassVar[int] = 32
    # Prompt tokens prefilled after the cached recurrent snapshot, which a next turn's template may render differently.
    uncached_prompt_tail: ClassVar[int] = 8


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"


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
type _DecodeCarry = tuple[State, Array, SamplingPolicy, Array]


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _pad_to_power_of_two[T](batch: list[T]) -> list[T]:
    # Padding rows repeat the first row, so their duplicate writes are identical.
    return batch + [batch[0]] * (_next_power_of_two(len(batch)) - len(batch))


def _prefill_capacity(prefix_length: int, head_width: int, tail_width: int) -> int:
    page_size = ContinuousBatchingConfig.page_size
    return math.ceil(_next_power_of_two(prefix_length + head_width + tail_width) / page_size) * page_size


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
) -> tuple[State, Array, Array, tuple[Array, Array, Array] | None, Array, SamplingPolicy]:
    def step(carry: _DecodeCarry, positions: Array) -> tuple[_DecodeCarry, tuple[Array, tuple[Array, ...] | None]]:
        state, logits, sampling_policy, sampling_keys = carry
        sample_keys = sampling_keys
        if not sampling_policy.is_greedy:
            sampling_keys, sample_keys = jnp.unstack(jax.vmap(jax.random.split)(sampling_keys), axis=1)
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
            logprobs = (token_logprobs, top_token_ids, top_logprobs)
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
        carry = (decoded.updated_state, decoded.logits[:, 0].astype(jnp.float32), sampling_policy, sampling_keys)
        return carry, (token_ids, logprobs)

    views = State(
        PagedKVCacheLayer(layer.keys, layer.values, block_tables, lengths)
        if isinstance(layer, PagedKVCachePool)
        else layer
        for layer in state
    )
    (views, logits, sampling_policy, sampling_keys), (token_ids, logprobs) = jax.lax.scan(
        step, (views, logits, sampling_policy, sampling_keys), token_positions
    )
    pools = State(
        PagedKVCachePool(layer.keys, layer.values) if isinstance(layer, PagedKVCacheLayer) else layer
        for layer in views
    )
    return pools, logits, token_ids, logprobs, sampling_keys, sampling_policy


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
            start = int(prefill.has_sinks)
            end = start + page_indices.shape[1] * pool.page_size
            layers.append(pool.write_pages(page_indices, prefill.keys[:, start:end], prefill.values[:, start:end]))
        else:
            layers.append(jax.tree.map(lambda old, new: old.at[slots].set(new), pool, prefill))
    return State(layers), logits.at[slots].set(prefilled_logits)


class CachedPrefix(NamedTuple):
    """KV pages and recurrent state left behind by a finished prompt, reusable by prompts that extend it."""

    token_ids: tuple[int, ...]
    pages: list[int]
    state_rows: tuple[StateLayerBase | None, ...]


@dataclass(eq=False)
class BatchingSequence:
    prompt_token_ids: tuple[int, ...]
    max_output_length: int
    stop_token_ids: tuple[int, ...]
    on_events: Callable[[Sequence[TokenEvent]], object]
    sampling_policy: SamplingPolicy
    sampling_key: Key[Array, ""]
    return_logprobs: bool
    cancelled: Event = field(default_factory=Event)
    completion_tokens: int = 0
    pages: list[int] = field(default_factory=list)
    slot: int | None = None
    cached_prefix: CachedPrefix | None = None
    head_length: int = 0
    head_state_rows: tuple[StateLayerBase | None, ...] | None = None

    @property
    def length(self) -> int:
        return len(self.prompt_token_ids) + self.completion_tokens


class ContinuousBatchingEngine:
    """Schedules sequences over a paged KV pool and per-slot recurrent state; `step` runs on a single worker thread."""

    def __init__(self, model: LanguageModel, config: ContinuousBatchingConfig) -> None:
        if model.sharding_config.resolve_axis(LogicalAxis.BATCH) is not None:
            raise ValueError("Continuous batching does not support batch-sharded models.")
        self.model = model
        self.config = config
        transformer = model.decoder.transformer
        attention_mixers = [layer.mixer for layer in transformer.layers if isinstance(layer.mixer, Attention)]
        devices = list(model.sharding_config.mesh.devices.flat)
        if attention_mixers and (
            any(device.platform != "gpu" for device in devices)
            or any(not mixer.config.is_causal for mixer in attention_mixers)
        ):
            raise ValueError("Paged batching requires causal GPU attention.")

        context_length = config.max_context_length
        if transformer.ropes:
            model_context = min(rope.config.max_sequence_length for rope in transformer.ropes)
            context_length = min(model_context, context_length or model_context)
        if context_length is None:
            raise ValueError("A model without a positional context limit requires max_context_length.")
        total_pages = config.total_pages if attention_mixers else 0
        if total_pages:
            # A sequence must fit in the pool on its own.
            context_length = min(context_length, total_pages * config.page_size)

        state_dtype = DecoderForwardPassConfig.for_inference().embedding_forward_pass_config.activation_dtype
        state_shape = jax.eval_shape(lambda: model.decoder.init_static_state(1, 1, state_dtype))
        page_bytes = config.page_size * sum(
            leaf.size // leaf.shape[1] * leaf.dtype.itemsize
            for layer in state_shape
            if isinstance(layer, StaticKVCacheLayer)
            for leaf in (layer.keys, layer.values)
        )
        recurrent_bytes = sum(
            leaf.size * leaf.dtype.itemsize
            for layer in state_shape
            if not isinstance(layer, StaticKVCacheLayer)
            for leaf in jax.tree.leaves(layer)
        )
        vocab_size = model.decoder.vocab_size
        pages_per_slot = math.ceil(context_length / config.page_size)
        slot_count = config.slot_count
        if total_pages:
            slot_count = min(slot_count or total_pages, total_pages // pages_per_slot)
        if total_pages is None or slot_count is None:
            stats = [device.memory_stats() or {} for device in devices]
            if not all(device_stats.get("bytes_limit") for device_stats in stats):
                raise ValueError("The device does not report its memory limit; configure total_pages and slot_count.")
            usable_bytes = int(0.9 * min(s["bytes_limit"] - s["bytes_in_use"] for s in stats))
            # Each slot keeps active state, a prefill snapshot, cached snapshots and padded decode rows, along with
            # persistent and working rows of logits and token-count penalties.
            slot_bytes = 8 * (recurrent_bytes + vocab_size * 4)
            # A worst-case prefill stages blank, cached-prefix and updated dense caches, with chunked attention scores.
            head_width = _next_power_of_two(max(1, context_length - 1 - config.uncached_prompt_tail))
            tail_width = _next_power_of_two(max(1, min(config.uncached_prompt_tail, context_length - 1)))
            prefill_capacity = _prefill_capacity(
                max(0, context_length - 1 - config.uncached_prompt_tail), head_width, tail_width
            )
            chunk_size = max(min(config.prefill_chunk_size, head_width), tail_width)
            num_heads = max((mixer.config.num_heads for mixer in attention_mixers), default=0)
            prefill_bytes = _next_power_of_two(config.prefill_batch_size) * (
                3 * prefill_capacity * page_bytes // config.page_size
                + 2 * recurrent_bytes
                + chunk_size * (2 * prefill_capacity * num_heads * 4 + vocab_size * 4)
            )
            # One extra page absorbs writes from padded batch rows and unallocated block-table entries.
            available_bytes = usable_bytes - prefill_bytes - page_bytes
            if slot_count is None:
                slot_count = available_bytes // (slot_bytes + pages_per_slot * page_bytes)
            if total_pages is None:
                total_pages = (available_bytes - slot_count * slot_bytes) // page_bytes
                slot_count = min(slot_count, total_pages // pages_per_slot)
            if slot_count < 1:
                raise ValueError("Not enough device memory for one sequence at the requested context length.")
            gib = total_pages * page_bytes / 2**30
            logger.info(
                "KV pool: %d pages (%.1f GiB), context %d, slots %d", total_pages, gib, context_length, slot_count
            )
        assert total_pages is not None
        self.context_limit = context_length
        self.slot_count = slot_count
        self.total_pages = total_pages

        static_state = model.decoder.init_static_state(slot_count, 1, state_dtype)
        layers = []
        for owner_index, static_layer in zip(transformer.kv_source_layer_indices, static_state, strict=True):
            mixer = transformer.layers[owner_index].mixer
            if not isinstance(mixer, Attention):
                layers.append(static_layer)
                continue
            shape = (mixer.config.num_groups, total_pages + 1, config.page_size, mixer.config.head_dim)
            sharding = model.sharding_config.make_sharding((None,) * len(shape))
            layers.append(
                PagedKVCachePool(
                    jnp.zeros(shape, dtype=state_dtype, device=sharding),
                    jnp.zeros(shape, dtype=state_dtype, device=sharding),
                )
            )
        self._state = State(layers)
        self._last_logits = jnp.zeros(
            (slot_count, vocab_size), dtype=jnp.float32, device=model.sharding_config.make_sharding((None, None))
        )
        self._incoming: SimpleQueue[BatchingSequence] = SimpleQueue()
        self._pending: deque[BatchingSequence] = deque()
        self._active: deque[BatchingSequence] = deque()
        self._free_slots: deque[int] = deque(range(slot_count))
        self._free_pages: deque[int] = deque(range(total_pages))
        self._prefix_cache: deque[CachedPrefix] = deque()

    def submit(
        self,
        prompt_token_ids: tuple[int, ...],
        max_output_length: int,
        generation_config: GenerationConfig,
        seed: int,
        *,
        return_logprobs: bool = False,
        on_events: Callable[[Sequence[TokenEvent]], object],
    ) -> Event:
        """Queues a sequence; setting the returned event cancels it."""
        assert prompt_token_ids and 0 < max_output_length <= self.context_limit - len(prompt_token_ids)
        sampling_policy = generation_config.default_policy()
        if sampling_policy.has_count_penalties:
            prompt = jnp.asarray(prompt_token_ids, dtype=jnp.int32)
            sampling_policy = sampling_policy.with_prompt_token_counts(
                prompt, jnp.asarray(len(prompt)), self.model.decoder.vocab_size
            )
        # JAX otherwise narrows Python seeds when x64 is disabled, discarding the upper 32 bits.
        with jax.enable_x64(new_val=True):
            sampling_key = jax.random.key(jnp.asarray(seed, dtype=jnp.int64))
        sequence = BatchingSequence(
            prompt_token_ids,
            max_output_length,
            generation_config.stop_token_ids,
            on_events,
            sampling_policy,
            sampling_key,
            return_logprobs,
        )
        self._incoming.put(sequence)
        return sequence.cancelled

    def step(self) -> bool:
        """Runs one prefill or decode block; returns False when there is no work."""
        while not self._incoming.empty():
            self._pending.append(self._incoming.get())
        self._pending = deque(sequence for sequence in self._pending if not sequence.cancelled.is_set())
        for sequence in tuple(self._active):
            if sequence.cancelled.is_set():
                self._active.remove(sequence)
                self._release(sequence)
        if self._prefill():
            return True
        if not self._active:
            return False
        self._decode()
        return True

    def _reserve_pages(self, sequence: BatchingSequence, new_tokens: int) -> None:
        # Slots are capped so that every active sequence can reach the context limit, so pages never run out.
        page_count = math.ceil((sequence.length + new_tokens) / self.config.page_size) - len(sequence.pages)
        if not self.total_pages or page_count <= 0:
            return
        while page_count > len(self._free_pages) and self._prefix_cache:
            self._free_pages.extend(self._prefix_cache.popleft().pages)
        assert page_count <= len(self._free_pages)
        sequence.pages.extend(self._free_pages.popleft() for _ in range(page_count))

    def _release(self, sequence: BatchingSequence) -> None:
        assert sequence.slot is not None
        self._free_slots.append(sequence.slot)
        sequence.slot = None
        if sequence.head_state_rows is not None:
            head_pages = math.ceil(sequence.head_length / self.config.page_size)
            self._prefix_cache.append(
                CachedPrefix(
                    sequence.prompt_token_ids[: sequence.head_length],
                    sequence.pages[:head_pages],
                    sequence.head_state_rows,
                )
            )
            sequence.pages = sequence.pages[head_pages:]
            if len(self._prefix_cache) > 2 * self.slot_count:
                self._free_pages.extend(self._prefix_cache.popleft().pages)
        self._free_pages.extend(sequence.pages)
        sequence.pages = []

    def _take_cached_prefix(self, prompt_token_ids: tuple[int, ...]) -> CachedPrefix | None:
        best = max(
            (
                prefix
                for prefix in self._prefix_cache
                if len(prefix.token_ids) < len(prompt_token_ids)
                and prompt_token_ids[: len(prefix.token_ids)] == prefix.token_ids
            ),
            key=lambda prefix: len(prefix.token_ids),
            default=None,
        )
        if best is not None:
            self._prefix_cache.remove(best)
        return best

    def _block_tables(self, page_lists: Sequence[list[int]], pages_per_sequence: int) -> Int[Array, "batch pages"]:
        padding = [self.total_pages] * pages_per_sequence
        return jnp.asarray([(pages + padding)[:pages_per_sequence] for pages in page_lists], dtype=jnp.int32)

    def _prefill(self) -> bool:
        batch: list[BatchingSequence] = []
        while self._pending and self._free_slots and len(batch) < self.config.prefill_batch_size:
            sequence = self._pending.popleft()
            sequence.cached_prefix = self._take_cached_prefix(sequence.prompt_token_ids)
            if sequence.cached_prefix is not None:
                sequence.pages = list(sequence.cached_prefix.pages)
            self._reserve_pages(sequence, 1)
            sequence.slot = self._free_slots.popleft()
            batch.append(sequence)
        if not batch:
            return False
        self._active.extend(batch)

        padded = _pad_to_power_of_two(batch)
        contexts = [sequence.prompt_token_ids for sequence in padded]
        prefix_lengths = [
            len(sequence.cached_prefix.token_ids) if sequence.cached_prefix else 0 for sequence in padded
        ]
        # Recurrent state is snapshotted a few tokens before the end, so that the next turn, whose template may
        # render the end of this prompt differently, can still extend the cached prefix.
        head_lengths = [
            max(prefix, len(context) - self.config.uncached_prompt_tail)
            for context, prefix in zip(contexts, prefix_lengths, strict=True)
        ]
        heads = [
            context[prefix:head] for context, prefix, head in zip(contexts, prefix_lengths, head_lengths, strict=True)
        ]
        tails = [context[head:] for context, head in zip(contexts, head_lengths, strict=True)]
        head_width = _next_power_of_two(max(1, *map(len, heads)))
        tail_width = _next_power_of_two(max(map(len, tails)))
        token_capacity = _prefill_capacity(max(prefix_lengths), head_width, tail_width)

        state = self._prefix_state(padded, prefix_lengths, token_capacity)
        if any(heads):
            chunk_size = min(self.config.prefill_chunk_size, head_width)
            state = self._prefill_rows(heads, head_width, chunk_size, token_capacity, state, prefix_lengths).state
        for row, sequence in enumerate(batch):
            sequence.cached_prefix = None
            sequence.head_length = head_lengths[row]
            sequence.head_state_rows = None
            if head_lengths[row] > 0:
                sequence.head_state_rows = tuple(
                    None
                    if isinstance(layer, StaticKVCacheLayer)
                    else jax.tree.map(lambda leaf, row=row: leaf[row], layer)
                    for layer in state
                )
        prefilled = self._prefill_rows(tails, tail_width, tail_width, token_capacity, state, head_lengths)
        self._state, self._last_logits = _merge_prefill(
            self._state,
            self._last_logits,
            prefilled.state,
            prefilled.last_token_logits,
            self._block_tables([sequence.pages for sequence in padded], token_capacity // self.config.page_size),
            jnp.asarray([sequence.slot for sequence in padded], dtype=jnp.int32),
        )
        return True

    def _prefill_rows(
        self,
        rows: list[tuple[int, ...]],
        width: int,
        chunk_size: int,
        token_capacity: int,
        state: State,
        prefix_lengths: list[int],
    ) -> PrefillResults:
        """Prefills each row, padded to `width`, after its row's first `prefix_lengths` tokens already in `state`."""
        tokens = np.zeros((len(rows), width), dtype=np.int32)
        for index, row in enumerate(rows):
            tokens[index, : len(row)] = row
        return self.model.prefill_tokens(
            jnp.asarray(tokens),
            token_capacity,
            jnp.asarray([len(row) for row in rows], dtype=jnp.int32),
            chunk_size=chunk_size,
            initial_state=state,
            prefix_lengths=jnp.asarray(prefix_lengths, dtype=jnp.int32),
            keychain=Keychain.init(0, sharding_config=self.model.sharding_config),
        )

    def _prefix_state(self, batch: list[BatchingSequence], prefix_lengths: list[int], token_capacity: int) -> State:
        """Static prefill state whose rows start with each sequence's cached prefix, or blank rows without one."""
        state_dtype = DecoderForwardPassConfig.for_inference().embedding_forward_pass_config.activation_dtype
        blank = self.model.decoder.init_static_state(len(batch), token_capacity, state_dtype)
        prefix_pages = math.ceil(max(prefix_lengths) / self.config.page_size)
        if prefix_pages == 0:
            return blank
        page_indices = self._block_tables(
            [sequence.cached_prefix.pages if sequence.cached_prefix else [] for sequence in batch], prefix_pages
        )
        layers = []
        for layer_index, (pool, layer) in enumerate(zip(self._state, blank, strict=True)):
            if isinstance(pool, PagedKVCachePool):
                assert isinstance(layer, StaticKVCacheLayer)
                keys, values = pool.read_pages(page_indices)
                start = int(layer.has_sinks)
                end = start + prefix_pages * self.config.page_size
                layers.append(
                    replace(
                        layer,
                        keys=layer.keys.at[:, start:end].set(keys.astype(layer.keys.dtype)),
                        values=layer.values.at[:, start:end].set(values.astype(layer.values.dtype)),
                        current_length=jnp.asarray(prefix_lengths, dtype=jnp.int32) + start,
                    )
                )
                continue
            seeded = layer
            for row, sequence in enumerate(batch):
                if sequence.cached_prefix is not None:
                    cached = sequence.cached_prefix.state_rows[layer_index]
                    seeded = jax.tree.map(
                        lambda leaf, cached_leaf, row=row: leaf.at[row].set(cached_leaf), seeded, cached
                    )
            layers.append(seeded)
        return State(layers)

    def _decode(self) -> None:
        # Sequences decode together when their sampling policies share a structure.
        first, *_ = self._active
        batch = [
            sequence
            for sequence in self._active
            if sequence.return_logprobs == first.return_logprobs
            and jax.tree.structure(sequence.sampling_policy) == jax.tree.structure(first.sampling_policy)
        ]
        block_size = min(64, *(sequence.max_output_length - sequence.completion_tokens for sequence in batch))
        block_size = 1 << (block_size.bit_length() - 1)
        for sequence in batch:
            self._reserve_pages(sequence, block_size)
        padded = _pad_to_power_of_two(batch)
        slots = jnp.asarray([sequence.slot for sequence in padded], dtype=jnp.int32)
        lengths = jnp.asarray([sequence.length for sequence in padded], dtype=jnp.int32)
        pages_per_sequence = 0
        if self.total_pages:
            pages_per_sequence = _next_power_of_two(max(len(sequence.pages) for sequence in padded))
        rows = State(
            layer if isinstance(layer, PagedKVCachePool) else jax.tree.map(lambda array: array[slots], layer)
            for layer in self._state
        )
        state, logits, token_ids, logprobs, sampling_keys, sampling_policies = _decode(
            self.model.decoder,
            rows,
            self._block_tables([sequence.pages for sequence in padded], pages_per_sequence),
            lengths,
            self._last_logits[slots],
            lengths[None, :, None] + jnp.arange(block_size, dtype=jnp.int32)[:, None, None],
            jax.tree.map(lambda *leaves: jnp.stack(leaves), *(sequence.sampling_policy for sequence in padded)),
            jnp.stack([sequence.sampling_key for sequence in padded]),
            first.return_logprobs,
        )
        self._state = State(
            new
            if isinstance(new, PagedKVCachePool)
            else jax.tree.map(lambda old, new: old.at[slots].set(new), old, new)
            for old, new in zip(self._state, state, strict=True)
        )
        self._last_logits = self._last_logits.at[slots].set(logits)

        token_ids = np.asarray(token_ids)
        host_logprobs = None if logprobs is None else [np.asarray(array) for array in logprobs]
        for row, sequence in enumerate(batch):
            self._active.remove(sequence)
            sequence.sampling_key = sampling_keys[row]
            sequence.sampling_policy = jax.tree.map(lambda leaf, row=row: leaf[row], sampling_policies)
            events: list[TokenEvent] = []
            for step, token_id in enumerate(map(int, token_ids[:, row])):
                sequence.completion_tokens += 1
                if token_id in sequence.stop_token_ids:
                    events.append(SequenceFinished(FinishReason.STOP, sequence.completion_tokens))
                    break
                selected_logprobs = None
                if host_logprobs is not None:
                    logprob, top_token_ids, top_logprobs = (array[step, row] for array in host_logprobs)
                    selected_logprobs = TokenLogprobs(
                        float(logprob), tuple(map(int, top_token_ids)), tuple(map(float, top_logprobs))
                    )
                events.append(GeneratedToken(token_id, selected_logprobs))
                if sequence.completion_tokens == sequence.max_output_length:
                    events.append(SequenceFinished(FinishReason.LENGTH, sequence.completion_tokens))
                    break
            if isinstance(events[-1], SequenceFinished):
                self._release(sequence)
            else:
                self._active.append(sequence)
            sequence.on_events(events)
