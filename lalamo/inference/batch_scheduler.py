import functools
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import batched

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Bool, DTypeLike, Float, Int, Key, Shaped

from lalamo.models.chat_codec import AssistantMessage, Message
from lalamo.models.language_model import DecodingState, GenerationConfig, LanguageModel, PrefillResults
from lalamo.module import ForwardPassMode, Keychain
from lalamo.modules import DecoderForwardPassConfig, State
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.common import Speculator

__all__ = [
    "BatchScheduler",
    "BatchSchedulerConfig",
    "BatchSchedulerKind",
    "BatchSizeInfo",
    "BatchSizesComputedEvent",
    "ContinuousBatchScheduler",
    "FixedSizeBatchScheduler",
    "GeneratedSequence",
    "estimate_batchsize_for_memory_budget",
]

type TokenSequence = list[int] | np.ndarray | jnp.ndarray

PROMPT_SIZE_BUCKETS: tuple[int, ...] = tuple(256 * 4**i for i in range(8))
MIN_BATCHES_PER_BUCKET: int = 10
MAX_BOUNDARY_BATCH_PADDING_FRACTION: float = 0.05


@dataclass(frozen=True)
class BatchSchedulerConfig:
    max_output_length: int = 8192
    padded_length: int | None = None
    batch_size: int | None = 1
    num_top_logits_to_return: int | None = None


@dataclass(frozen=True)
class BatchSizeInfo:
    prefix_length: int
    num_elements: int
    batch_size: int


@dataclass(frozen=True)
class BatchSizesComputedEvent:
    batch_sizes: tuple[BatchSizeInfo, ...]


@dataclass(frozen=True)
class GeneratedSequence:
    token_ids: Int[Array, " response_tokens"]
    top_k_token_ids: Int[Array, "response_tokens k"] | None = None
    top_k_token_logits: Float[Array, "response_tokens k"] | None = None


class BatchSchedulerKind(StrEnum):
    FIXED_SIZE = "fixed-size"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class DeviceMemoryStats:
    bytes_in_use: int
    bytes_limit: int


def _get_device_memory_stats() -> DeviceMemoryStats:
    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None:
        raise RuntimeError(
            "Automatic batch size estimation is not supported because device memory statistics are unavailable. "
            "Specify --batch-size explicitly.",
        )

    bytes_in_use = memory_stats.get("bytes_in_use")
    bytes_limit = memory_stats.get("bytes_limit")
    if bytes_in_use is None or bytes_limit is None:
        raise RuntimeError(
            "Automatic batch size estimation is not supported because device memory statistics are incomplete. "
            "Specify --batch-size explicitly.",
        )

    return DeviceMemoryStats(
        bytes_in_use=int(bytes_in_use),
        bytes_limit=int(bytes_limit),
    )


def _measure_peak_bytes_in_use(fn: Callable[[], None], poll_interval_ms: float = 5) -> int:
    peak_bytes_in_use = _get_device_memory_stats().bytes_in_use
    should_stop = threading.Event()
    polling_error = None

    def poll_memory() -> None:
        nonlocal peak_bytes_in_use, polling_error

        try:
            while not should_stop.is_set():
                peak_bytes_in_use = max(peak_bytes_in_use, _get_device_memory_stats().bytes_in_use)
                time.sleep(poll_interval_ms / 1000.0)
        except Exception as exc:  # noqa: BLE001
            polling_error = exc

    polling_thread = threading.Thread(target=poll_memory, daemon=True)
    polling_thread.start()
    try:
        fn()
    finally:
        should_stop.set()
        polling_thread.join()

    if polling_error is not None:
        raise RuntimeError("Memory polling thread failed.") from polling_error

    return peak_bytes_in_use


def _is_oom_error(error: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(error) or "out of memory" in str(error).lower()


def estimate_batchsize_for_memory_budget(
    fn: Callable[[int], None],
    memory_budget: int,
    num_steps: int,
    starting_batchsize: int = 2,
) -> int:
    batch_size = max(1, starting_batchsize)
    last_successful_batch_size: int | None = None
    remaining_steps = num_steps

    while True:
        try:
            peak_bytes_in_use = _measure_peak_bytes_in_use(functools.partial(fn, batch_size))
        except (JaxRuntimeError, ValueError) as error:
            if not _is_oom_error(error):
                raise

            next_batch_size = max(1, batch_size // 2)
            if next_batch_size == batch_size:
                if last_successful_batch_size is not None:
                    return last_successful_batch_size
                raise

            if last_successful_batch_size is not None:
                next_batch_size = max(last_successful_batch_size, next_batch_size)

            warnings.warn(
                f"OOM while estimating batch size at {batch_size}. Retrying with {next_batch_size}.",
                RuntimeWarning,
                stacklevel=2,
            )
            batch_size = next_batch_size
            continue

        last_successful_batch_size = batch_size
        estimated_batch_size = max(1, int(batch_size * memory_budget / peak_bytes_in_use))
        if estimated_batch_size <= batch_size:
            return estimated_batch_size
        if remaining_steps <= 0:
            return batch_size

        batch_size = estimated_batch_size
        remaining_steps -= 1


def _usable_bytes_from_available_bytes(available_bytes: int) -> int:
    return int(available_bytes * 0.98)


def _memory_budget_for_auto_batching(available_bytes: int) -> int:
    stats = _get_device_memory_stats()
    capped_bytes = min(available_bytes, stats.bytes_limit)
    if capped_bytes < available_bytes:
        warnings.warn(
            f"Clamping automatic batch-size memory budget from {available_bytes} to device limit {stats.bytes_limit}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _usable_bytes_from_available_bytes(capped_bytes)


def _assert_sorted(values: list[int]) -> None:
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1)), "expected sorted inputs"


def bucket_by_length[T: TokenSequence](tokenized: list[T]) -> dict[int, list[tuple[int, T]]]:
    if not tokenized:
        return {}
    max_length = max(len(sequence) for sequence in tokenized)
    bounds = [length for length in PROMPT_SIZE_BUCKETS if length < max_length] + [max_length]
    buckets: dict[int, list[tuple[int, T]]] = {}
    for idx, sequence in enumerate(tokenized):
        padded_length = min(length for length in bounds if length >= len(sequence))
        buckets.setdefault(padded_length, []).append((idx, sequence))
    return buckets


def bucket_sequences[T: TokenSequence](
    tokenized: list[T],
    memory_probe: Callable[[int, int], None],
    *,
    max_vram: int,
    starting_batch_size: int = 2,
    min_batches_per_bucket: int = MIN_BATCHES_PER_BUCKET,
) -> tuple[dict[int, list[tuple[int, T]]], dict[int, int]]:
    sequences_per_bucket = bucket_by_length(tokenized)
    sorted_lengths = sorted(sequences_per_bucket.keys(), reverse=True)
    batch_size_per_bucket: dict[int, int] = {}
    estimated = starting_batch_size
    num_steps = 3

    for idx, padded_length in enumerate(sorted_lengths):
        estimated = estimate_batchsize_for_memory_budget(
            functools.partial(memory_probe, padded_length=padded_length),
            memory_budget=max_vram,
            starting_batchsize=estimated,
            num_steps=num_steps,
        )

        # 1 estimator step suffices for each consecutive estimation, since we will already have
        # ~80% VRAM allocation ratio by starting from the last batchsize estimate, and 1 step is enough
        # to bring it to 98% ish util
        num_steps = 1

        batch_size_per_bucket[padded_length] = estimated
        remaining_lengths = sorted_lengths[idx + 1 :]

        # shortcircuit if there are only a few remaining samples -> fit all of them into the current bucket
        if sum(len(sequences_per_bucket[length]) for length in remaining_lengths) < min_batches_per_bucket * estimated:
            batch_size_per_bucket.update(dict.fromkeys(remaining_lengths, estimated))
            return sequences_per_bucket, batch_size_per_bucket

    return sequences_per_bucket, batch_size_per_bucket


def merge_small_buckets[T: TokenSequence](
    buckets: dict[int, list[tuple[int, T]]],
    batch_size_for_length: dict[int, int],
    min_batches: int = 4,
) -> dict[int, list[tuple[int, T]]]:
    # Given a bunch of buckets, this utility attempts to merge them upward in such a way that there is only one bucket
    # smaller that the required size of min_batches * batch_size.
    # Naturally, the "only" underfull bucket will be the largest one
    sorted_lengths = sorted(buckets.keys())
    merged: dict[int, list[tuple[int, T]]] = {}
    overflow: list[tuple[int, T]] = []

    for idx, padded_length in enumerate(sorted_lengths):
        batch_size = batch_size_for_length.get(padded_length, 1)
        items = overflow + buckets[padded_length]
        next_length = sorted_lengths[idx + 1] if idx + 1 < len(sorted_lengths) else padded_length

        if len(items) < min_batches * batch_size and next_length <= 2 * padded_length:
            overflow = items
        else:
            merged[padded_length] = items
            overflow = []

    if overflow:
        largest_length = sorted_lengths[-1]
        merged.setdefault(largest_length, []).extend(overflow)

    return merged


def pad_sequences(
    sequences: Iterable[TokenSequence],
    shape: tuple[int, int],
    *,
    dtype: DTypeLike,
    pad_value: int = 0,
) -> jnp.ndarray:
    batch_size, sequence_length = shape
    sequences_list = list(sequences)
    if len(sequences_list) > batch_size:
        raise RuntimeError(f"Expected at most {batch_size} sequences, got {len(sequences_list)}")

    padded = np.full((batch_size, sequence_length), pad_value, dtype=dtype)
    for idx, sequence in enumerate(sequences_list):
        sequence_array = np.asarray(sequence, dtype=dtype)
        if sequence_array.size > sequence_length:
            raise RuntimeError(f"Sequence length {sequence_array.size} exceeds target length {sequence_length}")
        padded[idx, : sequence_array.size] = sequence_array

    return jnp.asarray(padded)


def _decrease_batchsize_on_oom[T](
    fn: Callable[[int], Iterable[T]],
    starting_batch_size: int,
) -> Iterable[T]:
    first_batch_completed = False
    effective_batch_size = starting_batch_size

    while True:
        try:
            for result in fn(effective_batch_size):
                yield result
                first_batch_completed = True
            break
        except JaxRuntimeError as error:
            if first_batch_completed:
                raise
            new_batch_size = max(int(0.7 * effective_batch_size - 1), 1)
            if new_batch_size == 1 and effective_batch_size == 1:
                raise
            warnings.warn(
                f"OOM detected ({error}). Reducing batch size {effective_batch_size} -> {new_batch_size}.",
                RuntimeWarning,
                stacklevel=3,
            )
            effective_batch_size = new_batch_size


class PrefillBatch(eqx.Module):
    prefill_results: PrefillResults
    sampling_policy: SamplingPolicy
    mask: Bool[Array, " prefill_bs"]
    sampling_keys: Key[Array, " prefill_bs"]
    decoding_keys: Key[Array, " prefill_bs"]
    sequence_ids: Int[Array, " prefill_bs"]

    def __check_init__(self) -> None:
        (prefill_batch_size,) = self.prefill_results.last_token_indices.shape
        if self.mask.shape != (prefill_batch_size,):
            raise RuntimeError(f"mask shape {self.mask.shape} != ({prefill_batch_size},)")
        if self.sampling_keys.shape != (prefill_batch_size,):
            raise RuntimeError(f"sampling_keys shape {self.sampling_keys.shape} != ({prefill_batch_size},)")
        if self.decoding_keys.shape != (prefill_batch_size,):
            raise RuntimeError(f"decoding_keys shape {self.decoding_keys.shape} != ({prefill_batch_size},)")
        if self.sequence_ids.shape != (prefill_batch_size,):
            raise RuntimeError(f"sequence_ids shape {self.sequence_ids.shape} != ({prefill_batch_size},)")
        if self.prefill_results.last_token_logits.shape[0] != prefill_batch_size:
            raise RuntimeError(
                "prefill_results.last_token_logits batch "
                f"{self.prefill_results.last_token_logits.shape[0]} != {prefill_batch_size}",
            )


@dataclass(frozen=True)
class PrefillSource:
    tokenized_messages: list[list[int]]
    keychain: Keychain
    prefill_batch_size: int
    padded_length: int
    chunk_size: int
    state_capacity: int
    sampling_policy_template: SamplingPolicy
    cursor: int = 0

    @property
    def remaining(self) -> int:
        return len(self.tokenized_messages) - self.cursor

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.tokenized_messages)

    def fill(
        self,
        decoder: "BlockContinuousDecoder",
        state: "BlockContinuousState",
        jitted_fill_lines: Callable[["BlockContinuousState", "PrefillBatch"], "BlockContinuousState"],
    ) -> tuple["PrefillSource", "BlockContinuousState"]:
        source = self
        while state.empty_lines():
            if source.exhausted:
                break

            source, partially_prefilled_batch = source.request(len(state.empty_lines()), decoder.language_model)
            state = jitted_fill_lines(state, partially_prefilled_batch)
            del partially_prefilled_batch
        return source, state

    def request(self, num_requested: int, model: LanguageModel) -> tuple["PrefillSource", PrefillBatch]:
        num_designated = min(num_requested, self.prefill_batch_size, self.remaining)

        selected_indices = np.zeros(self.prefill_batch_size, dtype=np.int32)
        selected_indices[:num_designated] = np.arange(self.cursor, self.cursor + num_designated)

        sequences = self.tokenized_messages[self.cursor : self.cursor + num_designated]
        lengths_without_padding = np.zeros(self.prefill_batch_size, dtype=np.int32)
        lengths_without_padding[:num_designated] = [len(sequence) for sequence in sequences]
        padded_sequences = pad_sequences(sequences, (self.prefill_batch_size, self.padded_length), dtype=jnp.int32)
        mask = np.zeros(self.prefill_batch_size, dtype=bool)
        mask[:num_designated] = True

        selected_keys = self.keychain.vmapped_keys[jnp.asarray(selected_indices)]
        selected_keychain = Keychain(vmapped_keys=selected_keys, batch_key=self.keychain.batch_key)
        prefill_keychain, sampling_keychain, decoding_keychain = selected_keychain.split(3)
        prefilled_state = model.prefill_tokens(
            token_ids=padded_sequences,
            state_capacity=self.state_capacity,
            lengths_without_padding=jnp.asarray(lengths_without_padding),
            chunk_size=self.chunk_size,
            keychain=prefill_keychain,
        )
        sampling_policy = self.sampling_policy_template.broadcast(self.prefill_batch_size)
        # Avoid materializing batch x vocab token count storage unless a count-based penalty needs it.
        if self.sampling_policy_template.has_count_penalties:
            sampling_policy = call_vmapped(
                lambda policy, prompt_token_ids, prompt_length: policy.with_prompt_token_counts(
                    prompt_token_ids,
                    prompt_length,
                    model.decoder.vocab_size,
                ),
                sampling_policy,
                jnp.asarray(padded_sequences),
                jnp.asarray(lengths_without_padding),
            )

        batch = PrefillBatch(
            prefill_results=prefilled_state,
            sampling_policy=sampling_policy,
            mask=jnp.asarray(mask),
            sampling_keys=sampling_keychain.vmapped_keys,
            decoding_keys=decoding_keychain.vmapped_keys,
            sequence_ids=jnp.asarray(selected_indices),
        )
        return replace(self, cursor=self.cursor + num_designated), batch


def append_block_tokens(
    token_buffer: Int[Array, "num_lines max_output"],
    num_generated: Int[Array, " num_lines"],
    block_tokens: Int[Array, "block_size num_lines"],
) -> tuple[Int[Array, "num_lines max_output"], Int[Array, " num_lines"]]:
    num_lines, _ = token_buffer.shape
    block_size, _ = block_tokens.shape
    offsets = num_generated[:, None] + jnp.arange(block_size)
    new_buffer = token_buffer.at[jnp.arange(num_lines)[:, None], offsets].set(block_tokens.T, mode="drop")
    return new_buffer, num_generated + block_size


class BlockContinuousState(eqx.Module):
    last_token_logits: Float[Array, "num_lines vocab"]
    last_token_indices: Int[Array, " num_lines"]
    kv_state: State
    sampling_policy: SamplingPolicy
    stop_flags: Bool[Array, " num_lines"]
    token_buffer: Int[Array, "num_lines max_output"]
    num_generated: Int[Array, " num_lines"]
    sampling_keys: Key[Array, " num_lines"]
    decoding_keys: Key[Array, " num_lines"]
    in_batch_index_to_sequence_id: Int[Array, " num_lines"]

    def __check_init__(self) -> None:
        (num_lines,) = self.in_batch_index_to_sequence_id.shape
        for leaf in jax.tree.leaves(self):
            if leaf.ndim == 0:
                continue
            if leaf.shape[0] != num_lines:
                raise RuntimeError(f"BlockContinuousState leaf has first dim {leaf.shape[0]} != num_lines={num_lines}")

    @staticmethod
    def init(
        num_lines: int,
        max_output_length: int,
        state_capacity: int,
        model: LanguageModel,
        sampling_policy_template: SamplingPolicy,
    ) -> "BlockContinuousState":
        if sampling_policy_template.has_count_penalties:
            sampling_policy_template = sampling_policy_template.with_empty_token_counts(model.decoder.vocab_size)
        sampling_policy = sampling_policy_template.broadcast(num_lines)

        return BlockContinuousState(
            last_token_logits=jnp.zeros((num_lines, model.decoder.vocab_size), dtype=jnp.float32),
            last_token_indices=jnp.zeros(num_lines, dtype=jnp.int32),
            kv_state=model.decoder.init_static_state(
                num_lines,
                state_capacity,
                model.decoder.embedding.embedding_matrix.dtype,
            ),
            sampling_policy=sampling_policy,
            stop_flags=jnp.ones(num_lines, dtype=bool),
            token_buffer=jnp.zeros((num_lines, max_output_length), dtype=jnp.int32),
            num_generated=jnp.zeros(num_lines, dtype=jnp.int32),
            sampling_keys=jax.random.split(jax.random.key(0), num_lines),
            decoding_keys=jax.random.split(jax.random.key(1), num_lines),
            in_batch_index_to_sequence_id=jnp.zeros(num_lines, dtype=jnp.int32),
        )

    def empty_lines(self) -> list[int]:
        stopped = np.asarray(self.stop_flags)
        (num_lines,) = self.in_batch_index_to_sequence_id.shape
        return [idx for idx in range(num_lines) if stopped[idx]]

    def extract_completed_sequences(
        self,
        completed_mask: Bool[Array, " num_lines"],
    ) -> Iterator[tuple[int, GeneratedSequence]]:
        completed = np.asarray(completed_mask)
        sequence_ids = np.asarray(self.in_batch_index_to_sequence_id)
        for line in np.flatnonzero(completed):
            yield (
                int(sequence_ids[line]),
                GeneratedSequence(token_ids=self.token_buffer[line]),
            )

    def is_empty(self) -> bool:
        return bool(jnp.all(self.stop_flags))


class BlockContinuousDecoder(eqx.Module):
    block_size: int
    max_output_length: int
    language_model: LanguageModel
    sampling_policy_template: SamplingPolicy
    decoding_batch_key: Key[Array, ""]

    @classmethod
    def init(
        cls,
        max_output_length: int,
        block_size: int,
        model: LanguageModel,
        sampling_policy: SamplingPolicy,
        decoding_batch_key: Key[Array, ""],
    ) -> "BlockContinuousDecoder":
        return cls(
            block_size=block_size,
            max_output_length=max_output_length,
            language_model=model,
            sampling_policy_template=sampling_policy,
            decoding_batch_key=decoding_batch_key,
        )

    def _step(
        self,
        carry: tuple[DecodingState, Key[Array, " num_lines"], Key[Array, " num_lines"]],
        eos_token_ids: Int[Array, " eos_tokens"],
    ) -> tuple[tuple[DecodingState, Key[Array, " num_lines"], Key[Array, " num_lines"]], Int[Array, " num_lines"]]:
        decode_state, sampling_keys, decoding_keys = carry

        split_sampling_keys = jax.vmap(jax.random.split)(sampling_keys)
        next_sampling_keys = split_sampling_keys[:, 0]
        sample_keys = split_sampling_keys[:, 1]
        split_decoding_keys = jax.vmap(jax.random.split)(decoding_keys)
        next_decoding_keys = split_decoding_keys[:, 0]
        decoder_keys = split_decoding_keys[:, 1]

        processed_logits = call_vmapped(
            lambda policy, logits: policy.process_logits(logits),
            decode_state.sampling_policy,
            decode_state.last_token_logits.astype(jnp.float32),
        )
        next_token_ids = jax.vmap(jax.random.categorical)(sample_keys, processed_logits)
        next_token_ids = jnp.where(decode_state.stop_flags, 0, next_token_ids)
        next_sampling_policy = call_vmapped(
            lambda policy, token_id, should_count: policy.with_next_token_count(token_id, should_count),
            decode_state.sampling_policy,
            next_token_ids,
            jnp.logical_not(decode_state.stop_flags),
        )

        next_token_indices = decode_state.last_token_indices + 1
        stop_flags = jnp.logical_or(
            decode_state.stop_flags,
            jnp.any(next_token_ids[:, None] == eos_token_ids[None, :], axis=-1),
        )

        decoder_result = self.language_model.decoder(
            token_ids=next_token_ids[:, None],
            token_positions=next_token_indices[:, None],
            state=decode_state.state,
            return_updated_state=True,
            forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
            keychain=Keychain(vmapped_keys=decoder_keys, batch_key=self.decoding_batch_key),
        )
        assert decoder_result.updated_state is not None
        new_decode_state = DecodingState(
            last_token_logits=rearrange(decoder_result.logits, "num_lines 1 vocab -> num_lines vocab").astype(
                jnp.float32,
            ),
            last_token_indices=next_token_indices,
            state=decoder_result.updated_state,
            stop_flags=stop_flags,
            sampling_policy=next_sampling_policy,
        )
        return (new_decode_state, next_sampling_keys, next_decoding_keys), next_token_ids

    def fill_lines(self, state: BlockContinuousState, batch: PrefillBatch) -> BlockContinuousState:
        (prefill_batch_size,) = batch.prefill_results.last_token_indices.shape
        prefill_state = DecodingState(
            last_token_logits=batch.prefill_results.last_token_logits,
            last_token_indices=batch.prefill_results.last_token_indices,
            state=batch.prefill_results.state,
            stop_flags=jnp.zeros(prefill_batch_size, dtype=bool),
            sampling_policy=batch.sampling_policy,
        )
        current_decode_state = DecodingState(
            last_token_logits=state.last_token_logits,
            last_token_indices=state.last_token_indices,
            state=state.kv_state,
            stop_flags=state.stop_flags,
            sampling_policy=state.sampling_policy,
        )

        empty_lines = state.stop_flags
        lines_passed = jnp.sum(batch.mask)
        source_idx = jnp.cumsum(empty_lines, dtype=jnp.int32) - 1
        need_updating = jnp.logical_and(empty_lines, source_idx < lines_passed)

        def apply_updates(
            new: Shaped[Array, "num_lines ..."],
            old: Shaped[Array, "num_lines ..."],
        ) -> Shaped[Array, "num_lines ..."]:
            return jax.vmap(jnp.where)(need_updating, new, old)

        gathered_prefill_state = jax.tree.map(lambda value: value[source_idx], prefill_state)
        new_decode_state = jax.tree.map(apply_updates, gathered_prefill_state, current_decode_state)

        return BlockContinuousState(
            last_token_logits=new_decode_state.last_token_logits,
            last_token_indices=new_decode_state.last_token_indices,
            kv_state=new_decode_state.state,
            sampling_policy=new_decode_state.sampling_policy,
            stop_flags=new_decode_state.stop_flags,
            token_buffer=state.token_buffer,
            num_generated=apply_updates(jnp.zeros_like(state.num_generated), state.num_generated),
            sampling_keys=apply_updates(batch.sampling_keys[source_idx], state.sampling_keys),
            decoding_keys=apply_updates(batch.decoding_keys[source_idx], state.decoding_keys),
            in_batch_index_to_sequence_id=apply_updates(
                batch.sequence_ids[source_idx],
                state.in_batch_index_to_sequence_id,
            ),
        )

    def decode_block(self, state: BlockContinuousState) -> tuple[BlockContinuousState, Bool[Array, " num_lines"]]:
        eos_token_ids = jnp.asarray(self.language_model.config.generation_config.stop_token_ids, dtype=jnp.int32)
        initial_decode_state = DecodingState(
            last_token_logits=state.last_token_logits,
            last_token_indices=state.last_token_indices,
            state=state.kv_state,
            stop_flags=state.stop_flags,
            sampling_policy=state.sampling_policy,
        )
        (new_decode_state, new_sampling_keys, new_decoding_keys), block_tokens = jax.lax.scan(
            lambda carry, _: self._step(carry, eos_token_ids),
            (initial_decode_state, state.sampling_keys, state.decoding_keys),
            None,
            length=self.block_size,
        )

        new_buffer, new_num_generated = append_block_tokens(state.token_buffer, state.num_generated, block_tokens)
        combined_stop = jnp.logical_or(new_decode_state.stop_flags, new_num_generated >= self.max_output_length)
        completed_mask = jnp.logical_and(jnp.logical_not(state.stop_flags), combined_stop)

        return BlockContinuousState(
            last_token_logits=new_decode_state.last_token_logits,
            last_token_indices=new_decode_state.last_token_indices,
            kv_state=new_decode_state.state,
            sampling_policy=new_decode_state.sampling_policy,
            stop_flags=combined_stop,
            token_buffer=new_buffer,
            num_generated=new_num_generated,
            sampling_keys=new_sampling_keys,
            decoding_keys=new_decoding_keys,
            in_batch_index_to_sequence_id=state.in_batch_index_to_sequence_id,
        ), completed_mask


@dataclass(frozen=True)
class BatchScheduler(ABC):
    model: LanguageModel

    @abstractmethod
    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig | None = None,
        batch_scheduler_config: BatchSchedulerConfig = BatchSchedulerConfig(),
        *,
        fast_peak_memory: bool = False,
        speculator: Speculator | None = None,
        keychain: Keychain | None = None,
    ) -> Iterator[tuple[int, GeneratedSequence]]: ...

    def reply_many(
        self,
        messages: Iterable[Iterable[Message]],
        generation_config: GenerationConfig | None = None,
        batch_scheduler_config: BatchSchedulerConfig = BatchSchedulerConfig(),
        *,
        speculator: Speculator | None = None,
        enable_thinking: bool = True,
        keychain: Keychain | None = None,
        vram_bytes: int | None = None,
        batch_sizes_callback: Callable[[BatchSizesComputedEvent], None] | None = None,
    ) -> Iterator[tuple[int, AssistantMessage]]:
        messages = list(messages)
        keychain = (keychain or Keychain.init(0, shape=(len(messages),))).broadcast((len(messages),))

        if vram_bytes is not None and batch_scheduler_config.batch_size is not None:
            raise RuntimeError("Specify only one of batch_scheduler_config.batch_size and vram_bytes.")

        if vram_bytes is None and batch_scheduler_config.batch_size is None:
            raise RuntimeError("Specify either batch_scheduler_config.batch_size or vram_bytes.")

        tokenized = [
            self.model.token_codec.encode_request(message, enable_thinking=enable_thinking) for message in messages
        ]

        if batch_scheduler_config.batch_size is not None:
            if batch_scheduler_config.padded_length is None:
                sequences_per_bucket = bucket_by_length(tokenized)
            else:
                sequences_per_bucket = {
                    batch_scheduler_config.padded_length: list(enumerate(tokenized)),
                }
            batch_size_per_bucket = dict.fromkeys(sequences_per_bucket, batch_scheduler_config.batch_size)
        else:
            assert vram_bytes is not None
            first_token_of_the_first_sequence = tokenized[0][0]

            def memory_probe(batch_size: int, padded_length: int) -> None:
                iterator = self.generate_tokens_many(
                    [[first_token_of_the_first_sequence]] * batch_size,
                    generation_config=generation_config,
                    batch_scheduler_config=replace(
                        batch_scheduler_config,
                        batch_size=batch_size,
                        padded_length=padded_length,
                    ),
                    fast_peak_memory=True,
                    speculator=speculator,
                )
                first_result = next(iterator, None)
                if first_result is not None:
                    jax.block_until_ready(first_result)

            sequences_per_bucket, batch_size_per_bucket = bucket_sequences(
                tokenized,
                memory_probe,
                max_vram=_memory_budget_for_auto_batching(vram_bytes),
                min_batches_per_bucket=MIN_BATCHES_PER_BUCKET,
            )

        buckets = merge_small_buckets(sequences_per_bucket, batch_size_per_bucket, min_batches=MIN_BATCHES_PER_BUCKET)
        assert sum(len(bucket) for bucket in buckets.values()) == len(tokenized)
        capped_batch_size_per_bucket: dict[int, int] = {}
        for padded_length, bucket in buckets.items():
            batch_size = batch_size_per_bucket[padded_length]
            for candidate in range(min(batch_size, len(bucket)), 0, -1):
                num_slots = ((len(bucket) + candidate - 1) // candidate) * candidate
                if (num_slots - len(bucket)) / len(bucket) <= MAX_BOUNDARY_BATCH_PADDING_FRACTION:
                    break
            capped_batch_size_per_bucket[padded_length] = candidate
        batch_size_per_bucket = capped_batch_size_per_bucket

        if batch_sizes_callback is not None:
            batch_sizes = tuple(
                BatchSizeInfo(
                    prefix_length=padded_length,
                    num_elements=len(buckets[padded_length]),
                    batch_size=batch_size_per_bucket.get(padded_length, 1),
                )
                for padded_length in sorted(buckets)
            )
            batch_sizes_callback(BatchSizesComputedEvent(batch_sizes=batch_sizes))

        for padded_length in sorted(buckets, reverse=True):
            sequence_ids, sequence_tokenized = zip(*buckets[padded_length], strict=True)
            sequence_ids = list(sequence_ids)
            bucket_batch_size = batch_size_per_bucket[padded_length]
            bucket_config = replace(
                batch_scheduler_config,
                batch_size=bucket_batch_size,
                padded_length=padded_length,
            )
            bucket_keychain = Keychain(
                vmapped_keys=keychain.vmapped_keys[jnp.asarray(sequence_ids)],
                batch_key=keychain.batch_key,
            )

            for local_idx, result in self.generate_tokens_many(
                sequence_tokenized,
                generation_config=generation_config,
                batch_scheduler_config=bucket_config,
                fast_peak_memory=False,
                speculator=speculator,
                keychain=bucket_keychain,
            ):
                idx = sequence_ids[local_idx]
                trimmed_ids = self.model.trim_at_eos(result.token_ids.tolist())
                yield (idx, self.model.token_codec.decode_response(trimmed_ids, expect_thinking=enable_thinking))


@dataclass(frozen=True)
class FixedSizeBatchScheduler(BatchScheduler):
    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig | None = None,
        batch_scheduler_config: BatchSchedulerConfig = BatchSchedulerConfig(),
        *,
        fast_peak_memory: bool = False,
        speculator: Speculator | None = None,
        keychain: Keychain | None = None,
    ) -> Iterator[tuple[int, GeneratedSequence]]:
        tokenized = list(tokenized)
        if not tokenized:
            return
        if batch_scheduler_config.batch_size is None:
            raise RuntimeError("FixedSizeBatchScheduler requires batch_scheduler_config.batch_size.")

        keychain = (keychain or Keychain.init(0, shape=(len(tokenized),))).broadcast((len(tokenized),))

        def process_batches(batch_size: int) -> Iterator[tuple[int, GeneratedSequence]]:
            for batch_items in batched(enumerate(tokenized), batch_size):
                batch_with_ids = list(batch_items)
                batch_indices = [idx for idx, _ in batch_with_ids]
                real_batch = [tokens for _, tokens in batch_with_ids]
                padded_length = batch_scheduler_config.padded_length or max(len(tokens) for tokens in real_batch)
                num_padding_rows = batch_size - len(real_batch)
                lengths = jnp.asarray([len(tokens) for tokens in real_batch] + [1] * num_padding_rows, dtype=jnp.int32)
                padded = pad_sequences(real_batch, (batch_size, padded_length), dtype=jnp.int32)
                padded_batch_indices = batch_indices + [0] * num_padding_rows
                batch_keychain = Keychain(
                    vmapped_keys=keychain.vmapped_keys[jnp.asarray(padded_batch_indices)],
                    batch_key=keychain.batch_key,
                ).squeeze()
                batch_results = self.model.generate_tokens(
                    padded,
                    generation_config=generation_config,
                    prompt_lengths_without_padding=lengths,
                    max_output_length=batch_scheduler_config.max_output_length,
                    num_top_logits_to_return=batch_scheduler_config.num_top_logits_to_return,
                    speculator=speculator,
                    keychain=batch_keychain,
                )
                for local_idx, sequence_id in enumerate(batch_indices):
                    yield (
                        sequence_id,
                        GeneratedSequence(
                            token_ids=batch_results.token_ids[local_idx],
                            top_k_token_ids=(
                                batch_results.top_k_token_ids[local_idx]
                                if batch_results.top_k_token_ids is not None
                                else None
                            ),
                            top_k_token_logits=(
                                batch_results.top_k_token_logits[local_idx]
                                if batch_results.top_k_token_logits is not None
                                else None
                            ),
                        ),
                    )

        if fast_peak_memory:
            yield from process_batches(batch_scheduler_config.batch_size)
            return

        yield from _decrease_batchsize_on_oom(process_batches, starting_batch_size=batch_scheduler_config.batch_size)


@dataclass(frozen=True)
class ContinuousBatchScheduler(BatchScheduler):
    block_size: int = 256
    prefill_batch_fraction: float = 0.2
    prefill_chunk_size: int = 128

    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig | None = None,
        batch_scheduler_config: BatchSchedulerConfig = BatchSchedulerConfig(),
        *,
        fast_peak_memory: bool = False,
        speculator: Speculator | None = None,
        keychain: Keychain | None = None,
    ) -> Iterator[tuple[int, GeneratedSequence]]:
        if speculator is not None:
            raise NotImplementedError("ContinuousBatchScheduler does not support speculative decoding yet.")
        if batch_scheduler_config.num_top_logits_to_return is not None:
            raise RuntimeError("num_top_logits_to_return is not supported with ContinuousBatchScheduler.")

        tokenized = list(tokenized)
        if not tokenized:
            return
        if batch_scheduler_config.batch_size is None:
            raise RuntimeError("ContinuousBatchScheduler requires batch_scheduler_config.batch_size.")
        if batch_scheduler_config.padded_length is None:
            raise RuntimeError("ContinuousBatchScheduler requires batch_scheduler_config.padded_length.")

        keychain = (keychain or Keychain.init(0, shape=(len(tokenized),))).broadcast((len(tokenized),))

        sampling_policy = (
            generation_config.default_policy() if generation_config else self.model.default_sampling_policy()
        )

        batch_size = batch_scheduler_config.batch_size
        max_output_length = batch_scheduler_config.max_output_length
        padded_length = batch_scheduler_config.padded_length

        block_size = min(self.block_size, max_output_length)
        prefill_capacity = (
            (padded_length + self.prefill_chunk_size - 1) // self.prefill_chunk_size
        ) * self.prefill_chunk_size
        state_capacity = prefill_capacity + max_output_length + 1
        prefill_batch_size = max(1, min(int(batch_size * self.prefill_batch_fraction + 1), batch_size))

        prefills: PrefillSource = PrefillSource(
            tokenized_messages=tokenized,
            keychain=keychain,
            prefill_batch_size=prefill_batch_size,
            padded_length=padded_length,
            chunk_size=self.prefill_chunk_size,
            state_capacity=state_capacity,
            sampling_policy_template=sampling_policy,
        )

        # make sure the key used is the same as the one in FixedBatchScheduler
        _, _, decoding_keychain = keychain.split(3)
        decoder = BlockContinuousDecoder.init(
            max_output_length=max_output_length,
            block_size=block_size,
            model=self.model,
            sampling_policy=sampling_policy,
            decoding_batch_key=decoding_keychain.batch_key,
        )
        state = BlockContinuousState.init(
            num_lines=batch_size,
            max_output_length=max_output_length,
            state_capacity=state_capacity,
            model=self.model,
            sampling_policy_template=sampling_policy,
        )

        jitted_decode = eqx.filter_jit(decoder.decode_block, donate="all")
        jitted_fill_lines = eqx.filter_jit(decoder.fill_lines, donate="all")

        empty_lines = state.empty_lines()

        prefills, partially_prefilled_batch = prefills.request(len(empty_lines), decoder.language_model)

        jitted_fill_lines = jitted_fill_lines.lower(  # type: ignore[missing-attribute]
            state,
            partially_prefilled_batch,
        ).compile()
        state = jitted_fill_lines(state, partially_prefilled_batch)
        del partially_prefilled_batch

        prefills, state = prefills.fill(decoder, state, jitted_fill_lines)
        jitted_decode = jitted_decode.lower(state).compile()  # type: ignore[missing-attribute]

        while True:
            state, completed_mask = jitted_decode(state)
            yield from state.extract_completed_sequences(completed_mask)

            if state.is_empty() and prefills.exhausted:
                break

            if fast_peak_memory:
                break

            if not prefills.exhausted:
                prefills, state = prefills.fill(decoder, state, jitted_fill_lines)
