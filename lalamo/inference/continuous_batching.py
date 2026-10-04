import logging
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from queue import SimpleQueue
from threading import Event
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.models import GenerationConfig, LanguageModel
from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules import Decoder, DecoderForwardPassConfig, State
from lalamo.modules.token_mixer import StateLayerBase
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.kv_cache import PagedKVCacheLayer, PagedKVCachePool, StaticKVCacheLayer
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy
from lalamo.utils.memory import get_free_bytes

if TYPE_CHECKING:
    import xgrammar

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

    def __post_init__(self) -> None:
        for name in ("total_pages", "slot_count", "max_context_length", "prefill_batch_size", "prefill_chunk_size"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive.")


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


class GrammarConstraintError(ValueError):
    """The output constraint permits no next token in the model vocabulary."""


type TokenEvent = GeneratedToken | SequenceFinished | GrammarConstraintError
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


def _prefill_capacity(prefix_length: int, head_width: int, tail_width: int) -> int:
    page_size = ContinuousBatchingConfig.page_size
    return math.ceil(_next_power_of_two(prefix_length + head_width + tail_width) / page_size) * page_size


def _token_rows(rows: list[tuple[int, ...]], width: int) -> Int[Array, "batch width"]:
    tokens = np.zeros((len(rows), width), dtype=np.int32)
    for row, token_ids in enumerate(rows):
        tokens[row, : len(token_ids)] = token_ids
    return jnp.asarray(tokens)


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
            start = int(prefill.has_sinks)
            layers.append(
                pool.write_pages(
                    page_indices, prefill.keys[:, start : start + tokens], prefill.values[:, start : start + tokens]
                )
            )
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
    grammar_matcher: "xgrammar.GrammarMatcher | None" = None
    cancelled: Event = field(default_factory=Event)
    output_token_ids: list[int] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    slot: int | None = None
    cached_prefix: CachedPrefix | None = None
    head_length: int = 0
    head_state_rows: tuple[StateLayerBase | None, ...] | None = None

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
        attention_mixers = [layer.mixer for layer in transformer.layers if isinstance(layer.mixer, Attention)]
        if attention_mixers and (
            any(device.platform != "gpu" for device in model.sharding_config.mesh.devices.flat)
            or any(not mixer.config.is_causal for mixer in attention_mixers)
        ):
            raise ValueError("Paged batching requires causal GPU attention.")
        state_dtype = DecoderForwardPassConfig.for_inference().embedding_forward_pass_config.activation_dtype
        state_shape = jax.eval_shape(lambda: model.decoder.init_static_state(1, 1, state_dtype))
        page_bytes = None
        if attention_mixers:
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
        if transformer.ropes:
            model_context = min(rope.config.max_sequence_length for rope in transformer.ropes)
            context_length = model_context
            if config.max_context_length is not None:
                context_length = min(context_length, config.max_context_length)
        else:
            if config.max_context_length is None:
                raise ValueError("A model without a positional context limit requires max_context_length.")
            context_length = config.max_context_length
        total_pages = config.total_pages
        if page_bytes is None and total_pages is not None:
            raise ValueError("total_pages configures KV storage; this model has only recurrent state.")
        if total_pages is not None:
            context_length = min(config.page_size * total_pages, context_length)
        pages_per_slot = math.ceil(context_length / config.page_size)
        slot_count = config.slot_count
        if total_pages is not None:
            slots_that_fit = total_pages // pages_per_slot
            if slot_count is None:
                slot_count = slots_that_fit
            else:
                slot_count = min(slot_count, slots_that_fit)
        if (page_bytes is not None and total_pages is None) or config.slot_count is None:
            free_bytes = [get_free_bytes(device) for device in model.sharding_config.mesh.devices.flat]
            if any(value is None for value in free_bytes):
                raise ValueError("The device does not report free memory; configure pages and slots explicitly.")
            usable_bytes = int(min(value for value in free_bytes if value is not None) * 0.9)
            # State is replicated: active state, head snapshots, up to two cached snapshots per slot, and padded
            # decode rows. Logits and token-count penalties also need persistent and working vocabulary rows.
            slot_bytes = 8 * (recurrent_bytes + model.decoder.vocab_size * 4)
            head_width = _next_power_of_two(max(1, context_length - 9))
            tail_width = _next_power_of_two(max(1, min(8, context_length - 1)))
            prefill_capacity = _prefill_capacity(max(0, context_length - 9), head_width, tail_width)
            chunk_size = max(min(config.prefill_chunk_size, head_width), tail_width)
            num_heads = max((mixer.config.num_heads for mixer in attention_mixers), default=0)
            staging_bytes = 0
            if page_bytes is not None:
                # Blank staging, prefix reads and the updated dense cache can overlap before merging into the pool.
                staging_bytes = 3 * prefill_capacity * page_bytes // config.page_size
            prefill_bytes = (
                staging_bytes
                + 2 * recurrent_bytes
                + chunk_size * (2 * prefill_capacity * num_heads * 4 + model.decoder.vocab_size * 4)
            )

            def bytes_without_pool(slots: int) -> int:
                padded_batch = _next_power_of_two(min(slots, config.prefill_batch_size))
                return slots * slot_bytes + padded_batch * prefill_bytes

            minimum_pool_bytes = 0
            if page_bytes is not None:
                minimum_pool_bytes = page_bytes
                if total_pages is not None:
                    minimum_pool_bytes *= total_pages + 1
            upper = usable_bytes // slot_bytes
            if slot_count is not None:
                upper = min(upper, slot_count)
            lower = 0
            while lower < upper:
                candidate = (lower + upper + 1) // 2
                pool_bytes = minimum_pool_bytes
                if page_bytes is not None and total_pages is None:
                    pool_bytes += candidate * pages_per_slot * page_bytes
                if bytes_without_pool(candidate) + pool_bytes <= usable_bytes:
                    lower = candidate
                else:
                    upper = candidate - 1
            if lower == 0:
                raise ValueError("Not enough device memory for one sequence at the requested context length.")
            slot_count = lower
            if page_bytes is not None and total_pages is None:
                total_pages = (usable_bytes - bytes_without_pool(slot_count)) // page_bytes - 1
        assert slot_count is not None
        self.total_pages = total_pages
        self.context_limit = context_length
        self.slot_count = slot_count
        if total_pages is not None:
            assert page_bytes is not None
            logger.info(
                "KV pool: %d pages of %d tokens (%.1f GiB), context %d, slots %d",
                total_pages,
                config.page_size,
                total_pages * page_bytes / 2**30,
                self.context_limit,
                self.slot_count,
            )

        static_state = model.decoder.init_static_state(self.slot_count, 1, state_dtype)
        layers = []
        for owner_index, static_layer in zip(transformer.kv_source_layer_indices, static_state, strict=True):
            mixer = transformer.layers[owner_index].mixer
            if not isinstance(mixer, Attention):
                layers.append(static_layer)
                continue
            assert total_pages is not None
            # The extra page at index total_pages absorbs writes from padded batch rows and unallocated table entries.
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
            (self.slot_count, model.decoder.vocab_size),
            dtype=jnp.float32,
            device=model.sharding_config.make_sharding((None, None)),
        )
        self._incoming: SimpleQueue[BatchingSequence] = SimpleQueue()
        self._pending: deque[BatchingSequence] = deque()
        self._active: deque[BatchingSequence] = deque()
        self._free_slots: deque[int] = deque(range(self.slot_count))
        self._free_pages: deque[int] = deque()
        if total_pages is not None:
            self._free_pages.extend(range(total_pages))
        self._prefix_cache: deque[CachedPrefix] = deque()

    def submit(
        self,
        prompt_token_ids: tuple[int, ...],
        max_output_length: int,
        generation_config: GenerationConfig,
        seed: int,
        *,
        return_logprobs: bool = False,
        grammar_matcher: "xgrammar.GrammarMatcher | None" = None,
        on_events: Callable[[Sequence[TokenEvent]], object],
    ) -> Event:
        if not prompt_token_ids or max_output_length < 1:
            raise ValueError("A sequence requires a nonempty prompt and a positive output budget.")
        if len(prompt_token_ids) + max_output_length > self.context_limit:
            raise ValueError("The prompt and requested output exceed the configured context capacity.")
        sampling_policy = generation_config.default_policy(self.model.decoder.vocab_size)
        if grammar_matcher is not None:
            if set(grammar_matcher.stop_token_ids) != set(generation_config.stop_token_ids):
                raise ValueError("The grammar and generation must use the same stop tokens.")
            sampling_policy = replace(
                sampling_policy,
                allowed_token_bitmask=self._next_token_bitmask(grammar_matcher),
            )
        if sampling_policy.has_count_penalties:
            prompt = jnp.asarray(prompt_token_ids, dtype=jnp.int32)
            sampling_policy = sampling_policy.with_prompt_token_counts(
                prompt, jnp.asarray(len(prompt)), self.model.decoder.vocab_size
            )
        # JAX otherwise narrows Python seeds when x64 is disabled, discarding the upper 32 bits.
        with jax.enable_x64(new_val=True):
            sampling_key = jax.random.key(jnp.asarray(seed, dtype=jnp.int64))
        sequence = BatchingSequence(
            prompt_token_ids=prompt_token_ids,
            max_output_length=max_output_length,
            stop_token_ids=generation_config.stop_token_ids,
            on_events=on_events,
            sampling_policy=sampling_policy,
            sampling_key=sampling_key,
            return_logprobs=return_logprobs,
            grammar_matcher=grammar_matcher,
        )
        self._incoming.put(sequence)
        return sequence.cancelled

    def step(self) -> bool:
        while not self._incoming.empty():
            self._pending.append(self._incoming.get())
        self._pending = deque(sequence for sequence in self._pending if not sequence.cancelled.is_set())
        for sequence in tuple(self._active):
            if sequence.cancelled.is_set():
                self._active.remove(sequence)
                self._release(sequence, cache_prompt=True)
        if self._prefill():
            return True
        if not self._active:
            return False
        self._decode()
        return True

    def _available_pages(self) -> int:
        return len(self._free_pages) + sum(len(prefix.pages) for prefix in self._prefix_cache)

    def _reserve_pages(self, sequence: BatchingSequence, new_tokens: int) -> bool:
        if self.total_pages is None:
            return True
        page_count = math.ceil((sequence.length + new_tokens) / self.config.page_size) - len(sequence.pages)
        while page_count > len(self._free_pages) and self._prefix_cache:
            self._free_pages.extend(self._prefix_cache.popleft().pages)
        if page_count > len(self._free_pages):
            return False
        sequence.pages.extend(self._free_pages.popleft() for _ in range(page_count))
        return True

    def _release(self, sequence: BatchingSequence, *, cache_prompt: bool) -> None:
        assert sequence.slot is not None
        self._free_slots.append(sequence.slot)
        sequence.slot = None
        head_pages = math.ceil(sequence.head_length / self.config.page_size)
        if cache_prompt and sequence.head_state_rows is not None:
            context = (*sequence.prompt_token_ids, *sequence.output_token_ids)
            self._prefix_cache.append(
                CachedPrefix(context[: sequence.head_length], sequence.pages[:head_pages], sequence.head_state_rows)
            )
            sequence.pages = sequence.pages[head_pages:]
            if len(self._prefix_cache) > 2 * self.slot_count:
                self._free_pages.extend(self._prefix_cache.popleft().pages)
        self._free_pages.extend(sequence.pages)
        sequence.pages = []

    def _take_cached_prefix(self, prompt_token_ids: tuple[int, ...]) -> CachedPrefix | None:
        matches = [
            prefix
            for prefix in self._prefix_cache
            if len(prefix.token_ids) < len(prompt_token_ids)
            and prompt_token_ids[: len(prefix.token_ids)] == prefix.token_ids
        ]
        if not matches:
            return None
        best = max(matches, key=lambda prefix: len(prefix.token_ids))
        self._prefix_cache.remove(best)
        return best

    def _block_tables(self, batch: list[BatchingSequence], pages_per_sequence: int) -> Int[Array, "batch pages"]:
        if self.total_pages is None:
            return jnp.empty((len(batch), 0), dtype=jnp.int32)
        padding = [self.total_pages] * pages_per_sequence
        return jnp.asarray([(sequence.pages + padding)[:pages_per_sequence] for sequence in batch], dtype=jnp.int32)

    def _prefill(self) -> bool:
        batch: list[BatchingSequence] = []
        while self._pending and self._free_slots and len(batch) < self.config.prefill_batch_size:
            sequence = self._pending[0]
            if not sequence.output_token_ids:
                sequence.cached_prefix = self._take_cached_prefix(sequence.prompt_token_ids)
                sequence.pages = list(sequence.cached_prefix.pages) if sequence.cached_prefix else []
            if not self._reserve_pages(sequence, 1):
                if sequence.cached_prefix is not None:
                    self._prefix_cache.append(sequence.cached_prefix)
                    sequence.cached_prefix = None
                    sequence.pages = []
                break
            self._pending.popleft()
            sequence.slot = self._free_slots.popleft()
            batch.append(sequence)
        if not batch:
            return False
        self._active.extend(batch)

        padded = _pad_to_power_of_two(batch)
        contexts = [(*sequence.prompt_token_ids, *sequence.output_token_ids) for sequence in padded]
        prefix_lengths = [
            len(sequence.cached_prefix.token_ids) if sequence.cached_prefix else 0 for sequence in padded
        ]
        # The recurrent state is captured a few tokens before the end, so that the next turn, whose template renders
        # the tail of this prompt differently, can still extend the cached prefix.
        head_lengths = [
            max(prefix, len(context) - 8) for context, prefix in zip(contexts, prefix_lengths, strict=True)
        ]
        head_width = _next_power_of_two(
            max(1, *(head - prefix for head, prefix in zip(head_lengths, prefix_lengths, strict=False)))
        )
        tail_width = _next_power_of_two(
            max(len(context) - head for context, head in zip(contexts, head_lengths, strict=False))
        )
        token_capacity = _prefill_capacity(max(prefix_lengths), head_width, tail_width)
        pages_per_sequence = token_capacity // self.config.page_size
        keychain = Keychain.init(0, sharding_config=self.model.sharding_config)
        logger.info(
            "prefill batch=%d reused_prefix_tokens=%s new_tokens=%s",
            len(batch),
            prefix_lengths[: len(batch)],
            [len(context) - prefix for context, prefix in zip(contexts[: len(batch)], prefix_lengths, strict=False)],
        )

        state = self._prefix_state(padded, prefix_lengths, token_capacity)
        if any(head > prefix for head, prefix in zip(head_lengths, prefix_lengths, strict=True)):
            state = self.model.prefill_tokens(
                _token_rows(
                    [
                        context[prefix:head]
                        for context, prefix, head in zip(contexts, prefix_lengths, head_lengths, strict=False)
                    ],
                    head_width,
                ),
                token_capacity,
                jnp.asarray(
                    [head - prefix for head, prefix in zip(head_lengths, prefix_lengths, strict=True)], dtype=jnp.int32
                ),
                chunk_size=min(self.config.prefill_chunk_size, head_width),
                initial_state=state,
                prefix_lengths=jnp.asarray(prefix_lengths, dtype=jnp.int32),
                keychain=keychain,
            ).state
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
        prefilled = self.model.prefill_tokens(
            _token_rows([context[head:] for context, head in zip(contexts, head_lengths, strict=True)], tail_width),
            token_capacity,
            jnp.asarray(
                [len(context) - head for context, head in zip(contexts, head_lengths, strict=True)], dtype=jnp.int32
            ),
            chunk_size=tail_width,
            initial_state=state,
            prefix_lengths=jnp.asarray(head_lengths, dtype=jnp.int32),
            keychain=keychain,
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

    def _prefix_state(self, batch: list[BatchingSequence], prefix_lengths: list[int], token_capacity: int) -> State:
        """Static prefill state whose rows start with each sequence's cached prefix, or blank rows without one."""
        state_dtype = DecoderForwardPassConfig.for_inference().embedding_forward_pass_config.activation_dtype
        blank = self.model.decoder.init_static_state(len(batch), token_capacity, state_dtype)
        prefix_pages = math.ceil(max(prefix_lengths) / self.config.page_size)
        if prefix_pages == 0:
            return blank
        page_indices = self._block_tables(
            [
                replace(sequence, pages=sequence.cached_prefix.pages if sequence.cached_prefix else [])
                for sequence in batch
            ],
            prefix_pages,
        )
        layers = []
        for pool, layer in zip(self._state, blank, strict=True):
            if isinstance(pool, PagedKVCachePool):
                assert isinstance(layer, StaticKVCacheLayer)
                keys, values = pool.read_pages(page_indices)
                prefix_tokens = prefix_pages * self.config.page_size
                start = int(layer.has_sinks)
                layers.append(
                    replace(
                        layer,
                        keys=layer.keys.at[:, start : start + prefix_tokens].set(keys.astype(layer.keys.dtype)),
                        values=layer.values.at[:, start : start + prefix_tokens].set(
                            values.astype(layer.values.dtype)
                        ),
                        current_length=jnp.asarray(prefix_lengths, dtype=jnp.int32) + start,
                    )
                )
            else:
                seeded = layer
                for row, sequence in enumerate(batch):
                    if sequence.cached_prefix is not None:
                        cached = sequence.cached_prefix.state_rows[len(layers)]
                        seeded = jax.tree.map(
                            lambda leaf, row_leaf, row=row: leaf.at[row].set(row_leaf), seeded, cached
                        )
                layers.append(seeded)
        return State(layers)

    def _plan_decode(self) -> tuple[list[BatchingSequence], int] | None:
        first, *_ = self._active
        batch = [
            sequence
            for sequence in self._active
            if sequence.return_logprobs == first.return_logprobs
            and jax.tree.structure(sequence.sampling_policy) == jax.tree.structure(first.sampling_policy)
        ]
        block_size = min(64, *(sequence.remaining_output_length for sequence in batch))
        if first.grammar_matcher is not None:
            block_size = 1
        if self.total_pages is not None:
            growable_tokens = max(
                (len(sequence.pages) + self._available_pages()) * self.config.page_size - sequence.length
                for sequence in batch
            )
            block_size = min(block_size, growable_tokens)
        if block_size < 1:
            return None
        block_size = 1 << (block_size.bit_length() - 1)
        return [sequence for sequence in batch if self._reserve_pages(sequence, block_size)], block_size

    def _preempt(self) -> None:
        if len(self._active) == 1:
            raise RuntimeError("The sole active sequence cannot grow despite passing capacity validation.")
        victim = max(self._active, key=lambda sequence: len(sequence.pages))
        self._active.remove(victim)
        self._release(victim, cache_prompt=False)
        self._pending.appendleft(victim)

    def _decode(self) -> None:
        while (plan := self._plan_decode()) is None:
            self._preempt()
        batch, block_size = plan
        padded = _pad_to_power_of_two(batch)
        slots = jnp.asarray([sequence.slot for sequence in padded], dtype=jnp.int32)
        lengths = jnp.asarray([sequence.length for sequence in padded], dtype=jnp.int32)
        pages_per_sequence = 0
        if self.total_pages is not None:
            pages_per_sequence = _next_power_of_two(max(len(sequence.pages) for sequence in padded))
        block_tables = self._block_tables(padded, pages_per_sequence)
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
                if sequence.grammar_matcher is not None and not sequence.grammar_matcher.accept_token(token_id):
                    raise RuntimeError("A sampled token violated its grammar mask.")
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
                self._release(sequence, cache_prompt=True)
            else:
                try:
                    if sequence.grammar_matcher is not None:
                        sequence.sampling_policy = replace(
                            sequence.sampling_policy,
                            allowed_token_bitmask=self._next_token_bitmask(sequence.grammar_matcher),
                        )
                except GrammarConstraintError as error:
                    events.append(error)
                    self._release(sequence, cache_prompt=True)
                else:
                    self._active.append(sequence)
            sequence.on_events(events)

    def _next_token_bitmask(self, matcher: "xgrammar.GrammarMatcher") -> Int[Array, " vocabulary_words"]:
        bitmask = np.empty((1, math.ceil(self.model.decoder.vocab_size / 32)), dtype=np.int32)
        matcher.fill_next_token_bitmask(bitmask)
        remaining_bits = self.model.decoder.vocab_size % 32
        if remaining_bits:
            bitmask[0, -1] &= (1 << remaining_bits) - 1
        if not np.any(bitmask):
            raise GrammarConstraintError("The output constraint has no valid next token in the model vocabulary.")
        return jnp.asarray(bitmask[0])
