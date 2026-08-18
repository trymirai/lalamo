import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from threading import Event
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, Key, UInt

from lalamo.inference.page_allocator import PageAllocator
from lalamo.inference.paged_state import (
    _batch_sharding,
    _named_sharding,
    init_paged_state,
    insert_prefill,
    select_batch,
    update_slots,
)
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.language_model import _top_logits_with_remainder
from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules import Decoder, DecoderForwardPassConfig
from lalamo.modules.token_mixer import State
from lalamo.modules.token_mixers.attention import Attention
from lalamo.sampling import SamplingPolicy


@dataclass(frozen=True)
class ContinuousBatchingConfig:
    page_size: int = 32
    total_pages: int = 16_384
    slot_count: int = 64
    max_decode_batch_size: int = 64
    prefill_batch_size: int = 32
    prefill_chunk_size: int = 128
    decode_steps_per_prefill: int = 8
    decode_block_size: int = 512

    def __post_init__(self) -> None:
        for name in (
            "page_size",
            "total_pages",
            "slot_count",
            "max_decode_batch_size",
            "prefill_batch_size",
            "prefill_chunk_size",
            "decode_steps_per_prefill",
            "decode_block_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one.")
        if self.max_decode_batch_size > self.slot_count:
            raise ValueError("max_decode_batch_size cannot exceed slot_count.")
        if self.prefill_batch_size > self.slot_count:
            raise ValueError("prefill_batch_size cannot exceed slot_count.")


@dataclass(frozen=True)
class TokenizedRequest:
    request_id: str
    sequence_id: str
    prompt_token_ids: tuple[int, ...]
    max_output_length: int
    generation_config: GenerationConfig
    seed: int
    num_top_logits: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty.")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids cannot be empty.")
        if self.max_output_length < 1:
            raise ValueError("max_output_length must be at least one.")
        if self.num_top_logits is not None and self.num_top_logits < 1:
            raise ValueError("num_top_logits must be at least one.")


@dataclass(frozen=True)
class DecodeStepLogits:
    token_id: int
    top_token_ids: tuple[int, ...]
    top_raw_logits: tuple[float, ...]
    remainder_logit: float


@dataclass(frozen=True)
class CompletedRequest:
    request_id: str
    sequence_id: str
    output_token_ids: tuple[int, ...]
    logits: tuple[DecodeStepLogits, ...] | None
    error: str | None = None


@dataclass(frozen=True)
class ContinuousPrefillCompletedEvent:
    request_ids: tuple[str, ...]
    prompt_token_counts: tuple[int, ...]
    duration_seconds: float


@dataclass(frozen=True)
class ContinuousDecodeCompletedEvent:
    request_ids: tuple[str, ...]
    completed: tuple[bool, ...]
    duration_seconds: float


type ContinuousEngineEvent = ContinuousPrefillCompletedEvent | ContinuousDecodeCompletedEvent
type ContinuousEngineEventCallback = Callable[[ContinuousEngineEvent], None]


class PagedDecodeResult(NamedTuple):
    state: State
    last_logits: Float[Array, "slots vocabulary"]
    token_ids: Int[Array, " batch"]
    top_k_token_ids: Int[Array, "batch k"] | None
    top_k_token_logits: Float[Array, "batch k"] | None
    remainder_logits: Float[Array, " batch"] | None
    next_sampling_keys: Key[Array, " batch"] | None
    sampling_policy: SamplingPolicy | None


class _DecodeReadOnly(NamedTuple):
    decoder: Decoder
    slot_ids: Int[Array, " batch"]
    block_tables: Int[Array, "batch pages_per_sequence"]
    lengths: Int[Array, " batch"]
    sampling_policy: SamplingPolicy
    sampling_keys: Key[Array, " batch"]


class PagedDecodeBlockResult(NamedTuple):
    state: State
    last_logits: Float[Array, "slots vocabulary"]
    token_ids: Int[Array, "steps batch"]
    top_k_token_ids: Int[Array, "steps batch k"] | None
    top_k_token_logits: Float[Array, "steps batch k"] | None
    remainder_logits: Float[Array, "steps batch"] | None
    next_sampling_keys: Key[Array, " batch"]
    sampling_policy: SamplingPolicy


class _DecodeBlockCarry(NamedTuple):
    state: State
    last_logits: Float[Array, "slots vocabulary"]
    sampling_keys: Key[Array, " batch"]
    sampling_policy: SamplingPolicy


class _DecodeStepOutput(NamedTuple):
    token_ids: Int[Array, " batch"]
    top_k_token_ids: Int[Array, "batch k"] | None
    top_k_token_logits: Float[Array, "batch k"] | None
    remainder_logits: Float[Array, " batch"] | None


@eqx.filter_jit(donate="all-except-first")
def _decode_step(
    inputs: _DecodeReadOnly,
    state: State,
    last_logits: Float[Array, "slots vocabulary"],
    *,
    num_top_logits: int | None,
) -> PagedDecodeResult:
    decoder = inputs.decoder
    slot_ids = inputs.slot_ids
    block_tables = inputs.block_tables
    lengths = inputs.lengths
    sampling_policy = inputs.sampling_policy
    sampling_keys = inputs.sampling_keys
    selected_logits = last_logits.at[slot_ids].get(out_sharding=_batch_sharding(slot_ids, 2))
    token_ids, sampled_next_keys = jax.vmap(lambda policy, logits, key: policy.sample_with_next_key(logits, key))(
        sampling_policy,
        selected_logits,
        sampling_keys,
    )
    if sampling_policy.is_greedy:
        next_sampling_keys = None
    else:
        next_sampling_keys = sampled_next_keys
    if sampling_policy.has_count_penalties:
        updated_sampling_policy = jax.vmap(
            lambda policy, token_id: policy.with_next_token_count(token_id),
        )(sampling_policy, token_ids)
    else:
        updated_sampling_policy = None
    if num_top_logits is None:
        top_k_token_ids = None
        top_k_token_logits = None
        remainder_logits = None
    else:
        top_k_token_ids, top_k_token_logits, remainder_logits = _top_logits_with_remainder(
            selected_logits,
            num_top_logits,
        )
    decoder_result = decoder(
        token_ids[:, None],
        lengths[:, None],
        state=select_batch(state, slot_ids, block_tables, lengths),
        return_updated_state=True,
        forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
        # Inference forward passes draw no randomness; the keychain only satisfies the decoder signature.
        keychain=Keychain.init(0, sharding_config=decoder.sharding_config),
    )
    assert decoder_result.updated_state is not None
    return PagedDecodeResult(
        state=update_slots(state, decoder_result.updated_state, slot_ids),
        last_logits=last_logits.at[slot_ids].set(
            decoder_result.logits[:, 0].astype(jnp.float32),
            out_sharding=_named_sharding(last_logits),
        ),
        token_ids=token_ids,
        top_k_token_ids=top_k_token_ids,
        top_k_token_logits=top_k_token_logits,
        remainder_logits=remainder_logits,
        next_sampling_keys=next_sampling_keys,
        sampling_policy=updated_sampling_policy,
    )


@eqx.filter_jit(donate="all-except-first")
def _decode_block(
    inputs: _DecodeReadOnly,
    state: State,
    last_logits: Float[Array, "slots vocabulary"],
    *,
    block_size: int,
    num_top_logits: int | None,
) -> PagedDecodeBlockResult:
    def decode_iteration(
        carry: _DecodeBlockCarry,
        step: Int[Array, ""],
    ) -> tuple[_DecodeBlockCarry, _DecodeStepOutput]:
        result = _decode_step(
            inputs._replace(
                lengths=inputs.lengths + step,
                sampling_policy=carry.sampling_policy,
                sampling_keys=carry.sampling_keys,
            ),
            carry.state,
            carry.last_logits,
            num_top_logits=num_top_logits,
        )
        next_sampling_keys = carry.sampling_keys
        if result.next_sampling_keys is not None:
            next_sampling_keys = result.next_sampling_keys
        next_sampling_policy = carry.sampling_policy
        if result.sampling_policy is not None:
            next_sampling_policy = result.sampling_policy
        return (
            _DecodeBlockCarry(
                state=result.state,
                last_logits=result.last_logits,
                sampling_keys=next_sampling_keys,
                sampling_policy=next_sampling_policy,
            ),
            _DecodeStepOutput(
                token_ids=result.token_ids,
                top_k_token_ids=result.top_k_token_ids,
                top_k_token_logits=result.top_k_token_logits,
                remainder_logits=result.remainder_logits,
            ),
        )

    final, generated = jax.lax.scan(
        decode_iteration,
        _DecodeBlockCarry(state, last_logits, inputs.sampling_keys, inputs.sampling_policy),
        jnp.arange(block_size, dtype=jnp.int32),
    )
    return PagedDecodeBlockResult(
        state=final.state,
        last_logits=final.last_logits,
        token_ids=generated.token_ids,
        top_k_token_ids=generated.top_k_token_ids,
        top_k_token_logits=generated.top_k_token_logits,
        remainder_logits=generated.remainder_logits,
        next_sampling_keys=final.sampling_keys,
        sampling_policy=final.sampling_policy,
    )


@dataclass(eq=False)
class _RequestState:
    request: TokenizedRequest
    context_token_ids: list[int]
    sampling_policy: SamplingPolicy
    sampling_key: Key[Array, ""] | None = None
    logits: list[DecodeStepLogits] = field(default_factory=list)
    slot_id: int | None = None
    preemption_count: int = 0

    @classmethod
    def init(cls, request: TokenizedRequest, sampling_policy: SamplingPolicy) -> "_RequestState":
        return cls(
            request=request,
            context_token_ids=list(request.prompt_token_ids),
            sampling_policy=sampling_policy,
        )


class ContinuousBatchingEngine:
    def __init__(
        self,
        model: LanguageModel,
        config: ContinuousBatchingConfig,
        event_callback: ContinuousEngineEventCallback | None = None,
    ) -> None:
        if model.sharding_config.resolve_axis(LogicalAxis.BATCH) is not None:
            raise ValueError("Continuous batching does not support batch-sharded models.")
        self.model = model
        self.config = config
        self._event_callback = event_callback
        self._allocator = PageAllocator(total_pages=config.total_pages)
        state_dtype = DecoderForwardPassConfig.for_inference().embedding_forward_pass_config.activation_dtype
        self._state = init_paged_state(
            model.decoder,
            slot_count=config.slot_count,
            total_pages=config.total_pages + 1,
            page_size=config.page_size,
            dtype=state_dtype,
        )
        self._dummy_page_id = config.total_pages
        self._last_logits = jax.device_put(
            jnp.zeros((config.slot_count, model.decoder.vocab_size), dtype=jnp.float32),
            model.sharding_config.make_sharding((None, None)),
        )
        self._incoming: SimpleQueue[_RequestState] = SimpleQueue()
        self._pending: deque[_RequestState] = deque()
        self._active: deque[_RequestState] = deque()
        self._free_slot_ids: deque[int] = deque(range(config.slot_count))
        self._completed: SimpleQueue[CompletedRequest] = SimpleQueue()
        self._decode_steps_since_prefill = config.decode_steps_per_prefill
        self._keychain = Keychain.init(0, sharding_config=model.sharding_config)
        self._sampling_policies: dict[GenerationConfig, SamplingPolicy] = {}
        self._attention_sink_tokens = int(
            any(
                isinstance(layer.mixer, Attention) and layer.mixer.has_sinks
                for layer in model.decoder.transformer.layers
            )
        )

    def submit(self, request: TokenizedRequest) -> None:
        if request.num_top_logits is not None and request.num_top_logits >= self.model.decoder.vocab_size:
            raise ValueError(f"num_top_logits must be smaller than vocabulary size {self.model.decoder.vocab_size}.")
        sampling_policy = self._sampling_policies.get(request.generation_config)
        if sampling_policy is None:
            sampling_policy = request.generation_config.default_policy()
            self._sampling_policies[request.generation_config] = sampling_policy
        state = _RequestState.init(request, sampling_policy)
        if (
            len(state.context_token_ids) + self._attention_sink_tokens + 1
            > self.config.total_pages * self.config.page_size
        ):
            self._finish(state, error="Request context exceeds the paged KV-cache capacity.")
            return
        self._incoming.put(state)

    def pop_completed(self) -> CompletedRequest | None:
        try:
            return self._completed.get_nowait()
        except Empty:
            return None

    def run(self, stop_event: Event) -> None:
        while not stop_event.is_set():
            if not self.step():
                stop_event.wait(0.001)

    def step(self) -> bool:
        self._drain_incoming()
        should_prefill = bool(self._pending and self._free_slot_ids) and (
            len(self._active) < self.config.max_decode_batch_size
            or self._decode_steps_since_prefill >= self.config.decode_steps_per_prefill
        )
        if should_prefill and self._prefill():
            self._decode_steps_since_prefill = 0
            return True
        if decoded_steps := self._decode():
            self._decode_steps_since_prefill += decoded_steps
            return True
        if self._pending and self._prefill():
            self._decode_steps_since_prefill = 0
            return True
        while self._active:
            self._preempt_largest_request()
            if decoded_steps := self._decode():
                self._decode_steps_since_prefill += decoded_steps
                return True
        return False

    def _drain_incoming(self) -> None:
        while True:
            try:
                self._pending.append(self._incoming.get_nowait())
            except Empty:
                return

    def _prefill(self) -> bool:
        if not self._pending or not self._free_slot_ids:
            return False
        first = self._pending[0]
        required_page_count = math.ceil(
            (len(first.context_token_ids) + self._attention_sink_tokens + 1) / self.config.page_size
        )
        first_page_count = min(_bucket_size(required_page_count), self.config.total_pages)
        token_capacity = len(first.context_token_ids)
        if not self._attention_sink_tokens:
            token_capacity = _bucket_size(token_capacity)
        if first_page_count > len(self._allocator.free_page_ids):
            return False
        selected: list[_RequestState] = []
        selected_page_counts: list[int] = []
        selected_page_count = 0
        selected_limit = min(self.config.prefill_batch_size, len(self._free_slot_ids))
        pending_count = len(self._pending)
        for _ in range(pending_count):
            request = self._pending.popleft()
            request_token_capacity = len(request.context_token_ids)
            if not self._attention_sink_tokens:
                request_token_capacity = _bucket_size(request_token_capacity)
            if request_token_capacity != token_capacity:
                self._pending.append(request)
                continue
            required_page_count = math.ceil(
                (len(request.context_token_ids) + self._attention_sink_tokens + 1) / self.config.page_size
            )
            request_page_count = min(_bucket_size(required_page_count), self.config.total_pages)
            if selected_page_count + request_page_count > len(self._allocator.free_page_ids):
                self._pending.append(request)
                continue
            selected.append(request)
            selected_page_counts.append(request_page_count)
            selected_page_count += request_page_count
            if len(selected) == selected_limit:
                break

        slot_ids = tuple(self._free_slot_ids.popleft() for _ in selected)
        for request, request_page_count in zip(selected, selected_page_counts, strict=True):
            self._allocator.reserve(request.request.request_id, request_page_count)
        prefill_capacity = max(
            token_capacity,
            max(len(request.context_token_ids) for request in selected) + self._attention_sink_tokens,
        )
        pages_per_prefill_sequence = math.ceil(prefill_capacity / self.config.page_size)
        batch_size = len(selected)
        compiled_batch_size = _bucket_size(batch_size)
        padded_selected = [*selected, *([first] * (compiled_batch_size - batch_size))]
        padded_slot_ids = (*slot_ids, *((slot_ids[0],) * (compiled_batch_size - batch_size)))
        page_ids_by_request = tuple(
            self._allocator.pages_for(request.request.request_id) for request in padded_selected
        )
        block_tables = tuple(
            (
                *page_ids[:pages_per_prefill_sequence],
                *((self._dummy_page_id,) * max(pages_per_prefill_sequence - len(page_ids), 0)),
            )
            for page_ids in page_ids_by_request
        )
        state_capacity = pages_per_prefill_sequence * self.config.page_size
        token_ids = np.zeros((compiled_batch_size, token_capacity), dtype=np.int32)
        self._initialize_sampling_keys(selected)
        for row, request in enumerate(padded_selected):
            token_ids[row, : len(request.context_token_ids)] = request.context_token_ids
            policy = request.sampling_policy
            if policy.has_count_penalties and policy.token_counts is None:
                request.sampling_policy = policy.with_prompt_token_counts(
                    jnp.asarray(request.context_token_ids, dtype=jnp.int32),
                    jnp.asarray(len(request.context_token_ids), dtype=jnp.int32),
                    self.model.decoder.vocab_size,
                )
        lengths = np.asarray([len(request.context_token_ids) for request in padded_selected], dtype=np.int32)
        batch_token_sharding = self.model.sharding_config.make_sharding((None, None))
        batch_vector_sharding = self.model.sharding_config.make_sharding((None,))
        token_ids_array = jax.device_put(jnp.asarray(token_ids), batch_token_sharding)
        lengths_array = jax.device_put(jnp.asarray(lengths), batch_vector_sharding)
        slot_ids_array = jax.device_put(jnp.asarray(padded_slot_ids, dtype=jnp.int32), batch_vector_sharding)
        block_tables_array = jax.device_put(jnp.asarray(block_tables, dtype=jnp.int32), batch_token_sharding)
        started_at_seconds = time.perf_counter()
        prefill_result = self.model.prefill_tokens(
            token_ids_array,
            state_capacity=state_capacity,
            lengths_without_padding=lengths_array,
            chunk_size=self.config.prefill_chunk_size,
            keychain=self._keychain,
        )
        self._state = insert_prefill(
            self._state,
            prefill_result.state,
            slot_ids_array,
            block_tables_array,
        )
        self._last_logits = self._last_logits.at[slot_ids_array].set(
            prefill_result.last_token_logits,
            out_sharding=_named_sharding(self._last_logits),
        )
        if self._event_callback is not None:
            jax.block_until_ready(self._last_logits)
            self._event_callback(
                ContinuousPrefillCompletedEvent(
                    request_ids=tuple(request.request.request_id for request in selected),
                    prompt_token_counts=tuple(len(request.request.prompt_token_ids) for request in selected),
                    duration_seconds=time.perf_counter() - started_at_seconds,
                )
            )
        for request, slot_id in zip(selected, slot_ids, strict=True):
            request.slot_id = slot_id
            self._active.append(request)
        return True

    def _initialize_sampling_keys(self, requests: list[_RequestState]) -> None:
        missing = [request for request in requests if request.sampling_key is None]
        if not missing:
            return
        seed_count = self.config.prefill_batch_size
        seeds = np.zeros(seed_count, dtype=np.uint32)
        seeds[: len(missing)] = [request.request.seed % 2**32 for request in missing]
        sampling_keys = _initial_sampling_keys(jnp.asarray(seeds))
        for request, sampling_key in zip(missing, sampling_keys, strict=False):
            request.sampling_key = sampling_key

    def _decode(self) -> int:
        if not self._active:
            return 0
        cache_capacity = self.config.total_pages * self.config.page_size - self._attention_sink_tokens
        block_size = min(
            self.config.decode_block_size,
            *(cache_capacity - len(request.context_token_ids) for request in self._active),
            *(
                request.request.max_output_length
                - (len(request.context_token_ids) - len(request.request.prompt_token_ids))
                for request in self._active
            ),
        )
        selected = self._select_decode_batch(block_size)
        if not selected:
            return 0
        first = selected[0]
        num_top_logits = first.request.num_top_logits
        batch_size = _bucket_size(len(selected))
        padded_selected = [*selected, *([first] * (batch_size - len(selected)))]
        pages_by_request = [self._allocator.pages_for(request.request.request_id) for request in padded_selected]
        pages_per_sequence = _bucket_size(max(len(page_ids) for page_ids in pages_by_request))
        slot_ids = jnp.asarray([self._slot_id(request) for request in padded_selected], dtype=jnp.int32)
        block_tables = jnp.asarray(
            [
                (*page_ids, *((self._dummy_page_id,) * (pages_per_sequence - len(page_ids))))
                for page_ids in pages_by_request
            ],
            dtype=jnp.int32,
        )
        lengths = jnp.asarray([len(request.context_token_ids) for request in padded_selected], dtype=jnp.int32)
        sampling_policy = jax.tree.map(
            lambda leaf: jax.device_put(
                leaf,
                self.model.sharding_config.make_sharding((None,) * leaf.ndim),
            ),
            _stack_sampling_policies(padded_selected),
        )
        sampling_keys = jax.device_put(
            jnp.stack([_sampling_key(request) for request in padded_selected]),
            self.model.sharding_config.make_sharding((None,)),
        )
        started_at_seconds = time.perf_counter()
        result = _decode_block(
            _DecodeReadOnly(
                decoder=self.model.decoder,
                slot_ids=slot_ids,
                block_tables=block_tables,
                lengths=lengths,
                sampling_policy=sampling_policy,
                sampling_keys=sampling_keys,
            ),
            self._state,
            self._last_logits,
            block_size=block_size,
            num_top_logits=num_top_logits,
        )
        self._state = result.state
        self._last_logits = result.last_logits
        token_ids = np.asarray(result.token_ids)[:, : len(selected)]
        if num_top_logits is None:
            top_token_ids = None
            top_logits = None
            remainder_logits = None
        else:
            assert result.top_k_token_ids is not None
            assert result.top_k_token_logits is not None
            assert result.remainder_logits is not None
            top_token_ids = np.asarray(result.top_k_token_ids)[:, : len(selected)]
            top_logits = np.asarray(result.top_k_token_logits)[:, : len(selected)]
            remainder_logits = np.asarray(result.remainder_logits)[:, : len(selected)]
        for row, request in enumerate(selected):
            request.sampling_key = result.next_sampling_keys[row]
            request.sampling_policy = jax.tree.map(lambda leaf, row=row: leaf[row], result.sampling_policy)
        decode_step_duration_seconds = (time.perf_counter() - started_at_seconds) / block_size
        event_request_ids: list[list[str]] = [[] for _ in range(block_size)]
        event_completed: list[list[bool]] = [[] for _ in range(block_size)]
        for row, request in enumerate(selected):
            completed = False
            for step in range(block_size):
                generated_token_id = int(token_ids[step, row])
                request.context_token_ids.append(generated_token_id)
                if num_top_logits is not None:
                    assert top_token_ids is not None
                    assert top_logits is not None
                    assert remainder_logits is not None
                    request.logits.append(
                        DecodeStepLogits(
                            token_id=generated_token_id,
                            top_token_ids=tuple(int(value) for value in top_token_ids[step, row]),
                            top_raw_logits=tuple(float(value) for value in top_logits[step, row]),
                            remainder_logit=float(remainder_logits[step, row]),
                        )
                    )
                completed = (
                    generated_token_id in self.model.config.generation_config.stop_token_ids
                    or len(request.context_token_ids) - len(request.request.prompt_token_ids)
                    >= request.request.max_output_length
                )
                event_request_ids[step].append(request.request.request_id)
                event_completed[step].append(completed)
                if completed:
                    break
            if completed:
                self._release(request)
                self._finish(request)
            else:
                self._active.remove(request)
                self._active.append(request)

        if self._event_callback is not None:
            for request_ids, completed_flags in zip(event_request_ids, event_completed, strict=True):
                if not request_ids:
                    continue
                self._event_callback(
                    ContinuousDecodeCompletedEvent(
                        request_ids=tuple(request_ids),
                        completed=tuple(completed_flags),
                        duration_seconds=decode_step_duration_seconds,
                    )
                )
        return block_size

    def _select_decode_batch(self, block_size: int) -> list[_RequestState]:
        for request in tuple(self._active):
            if (
                len(request.context_token_ids) + self._attention_sink_tokens + 1
                > self.config.total_pages * self.config.page_size
            ):
                self._release(request)
                self._finish(request, error="Request context reached the paged KV-cache capacity.")

        selected: list[_RequestState] = []
        for request in self._active:
            if selected and (
                request.request.num_top_logits != selected[0].request.num_top_logits
                or request.request.generation_config != selected[0].request.generation_config
            ):
                continue
            if self._ensure_decode_capacity(request, block_size):
                selected.append(request)
                if len(selected) == self.config.max_decode_batch_size:
                    break
        return selected

    def _ensure_decode_capacity(self, request: _RequestState, block_size: int) -> bool:
        request_id = request.request.request_id
        required_page_count = math.ceil(
            (len(request.context_token_ids) + self._attention_sink_tokens + block_size) / self.config.page_size
        )
        return self._allocator.grow(request_id, required_page_count)

    def _preempt_largest_request(self) -> None:
        victim = min(
            self._active,
            key=lambda request: (
                request.preemption_count,
                -len(self._allocator.pages_for(request.request.request_id)),
            ),
        )
        self._release(victim)
        victim.preemption_count += 1
        self._pending.append(victim)

    def _release(self, request: _RequestState) -> None:
        request_id = request.request.request_id
        self._allocator.release(request_id)
        self._free_slot_ids.append(self._slot_id(request))
        self._active.remove(request)
        request.slot_id = None

    def _finish(self, request: _RequestState, error: str | None = None) -> None:
        prompt_length = len(request.request.prompt_token_ids)
        self._completed.put(
            CompletedRequest(
                request_id=request.request.request_id,
                sequence_id=request.request.sequence_id,
                output_token_ids=tuple(request.context_token_ids[prompt_length:]),
                logits=None if request.request.num_top_logits is None else tuple(request.logits),
                error=error,
            )
        )

    @staticmethod
    def _slot_id(request: _RequestState) -> int:
        if request.slot_id is None:
            raise RuntimeError(f"Request {request.request.request_id!r} has no active slot.")
        return request.slot_id


def _bucket_size(size: int) -> int:
    return 1 << (size - 1).bit_length()


def _stack_sampling_policies(requests: list[_RequestState]) -> SamplingPolicy:
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *(request.sampling_policy for request in requests))


@jax.jit
def _initial_sampling_keys(seeds: UInt[Array, " batch"]) -> Key[Array, " batch"]:
    def initial_sampling_key(seed: UInt[Array, ""]) -> Key[Array, ""]:
        initial_key, _ = jax.random.split(jax.random.key(seed))
        _, sampling_key, _ = jax.random.split(initial_key, 3)
        _, first_step_key = jax.random.split(sampling_key)
        return first_step_key

    return jax.vmap(initial_sampling_key)(seeds)


def _sampling_key(request: _RequestState) -> Key[Array, ""]:
    assert request.sampling_key is not None
    return request.sampling_key
