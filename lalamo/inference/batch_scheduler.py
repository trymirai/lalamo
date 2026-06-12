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
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Bool, DTypeLike, Float, Int, Key, Shaped

from lalamo.models.chat_codec import AssistantMessage, Message
from lalamo.models.language_model import DecodingState, GenerationConfig, LanguageModel, PrefillResults
from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules import DecoderForwardPassConfig
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy
from lalamo.speculator import NoSpeculator, Speculator

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
    return int(available_bytes * 0.9)


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
    batch_size_multiple: int = 1,
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
            smallest_batch_size = batch_size_multiple
            if effective_batch_size == smallest_batch_size:
                raise
            raw_new_batch_size = max(int(0.7 * effective_batch_size - 1), smallest_batch_size)
            new_batch_size = max(
                (raw_new_batch_size // batch_size_multiple) * batch_size_multiple,
                smallest_batch_size,
            )
            if new_batch_size >= effective_batch_size:
                new_batch_size = effective_batch_size - batch_size_multiple
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
    speculator: Speculator
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
        batch_axis = model.sharding_config.resolve_axis(LogicalAxis.BATCH)
        batch_token_sharding = model.sharding_config.make_sharding((batch_axis, None))
        batch_vector_sharding = model.sharding_config.make_sharding((batch_axis,))

        selected_indices = np.zeros(self.prefill_batch_size, dtype=np.int32)
        selected_indices[:num_designated] = np.arange(self.cursor, self.cursor + num_designated)
        selected_indices_array = jnp.asarray(selected_indices)

        sequences = self.tokenized_messages[self.cursor : self.cursor + num_designated]
        lengths_without_padding = np.zeros(self.prefill_batch_size, dtype=np.int32)
        lengths_without_padding[:num_designated] = [len(sequence) for sequence in sequences]
        padded_sequences = jax.device_put(
            pad_sequences(sequences, (self.prefill_batch_size, self.padded_length), dtype=jnp.int32),
            batch_token_sharding,
        )
        lengths_without_padding_array = jax.device_put(jnp.asarray(lengths_without_padding), batch_vector_sharding)
        mask = np.zeros(self.prefill_batch_size, dtype=bool)
        mask[:num_designated] = True
        mask_array = jax.device_put(jnp.asarray(mask), batch_vector_sharding)
        sequence_ids = jax.device_put(selected_indices_array, batch_vector_sharding)

        selected_keys = self.keychain.vmapped_keys.at[selected_indices_array].get(out_sharding=batch_vector_sharding)
        selected_keychain = Keychain(
            vmapped_keys=selected_keys,
            batch_key=self.keychain.batch_key,
            sharding_config=self.keychain.sharding_config,
        )
        prefill_keychain, sampling_keychain, decoding_keychain = selected_keychain.split(3)
        prefilled_state = model.prefill_tokens(
            token_ids=padded_sequences,
            state_capacity=self.state_capacity,
            lengths_without_padding=lengths_without_padding_array,
            chunk_size=self.chunk_size,
            speculator=self.speculator,
            keychain=prefill_keychain,
        )
        sampling_policy = self.sampling_policy_template.broadcast(self.prefill_batch_size)

        def shard_sampling_leaf(leaf: object) -> object:
            if not isinstance(leaf, jax.Array):
                return leaf
            if leaf.ndim > 0 and leaf.shape[0] == self.prefill_batch_size:
                sharding_axes = (batch_axis, *((None,) * (leaf.ndim - 1)))
            else:
                sharding_axes = (None,) * leaf.ndim
            leaf_sharding = model.sharding_config.make_sharding(sharding_axes)
            return jax.device_put(leaf, leaf_sharding)

        sampling_policy = jax.tree.map(shard_sampling_leaf, sampling_policy)
        # Avoid materializing batch x vocab token count storage unless a count-based penalty needs it.
        if self.sampling_policy_template.has_count_penalties:
            sampling_policy = call_vmapped(
                lambda policy, prompt_token_ids, prompt_length: policy.with_prompt_token_counts(
                    prompt_token_ids,
                    prompt_length,
                    model.decoder.vocab_size,
                ),
                sampling_policy,
                padded_sequences,
                lengths_without_padding_array,
            )

        batch = PrefillBatch(
            prefill_results=prefilled_state,
            sampling_policy=sampling_policy,
            mask=mask_array,
            sampling_keys=jax.device_put(sampling_keychain.vmapped_keys, batch_vector_sharding),
            decoding_keys=jax.device_put(decoding_keychain.vmapped_keys, batch_vector_sharding),
            sequence_ids=sequence_ids,
        )
        return replace(self, cursor=self.cursor + num_designated), batch


class BlockContinuousState(eqx.Module):
    decoding_state: DecodingState
    token_buffer: Int[Array, "num_lines max_output"]
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
    def from_decoding_state(
        prefill_decoding_state: DecodingState,
        num_lines: int,
        max_output_length: int,
        key: Key[Array, ""],
    ) -> "BlockContinuousState":
        def empty_line_values(leaf: Shaped[Array, "prefill_bs ..."]) -> Shaped[Array, "num_lines ..."]:
            _, *line_dims = leaf.shape
            return jnp.zeros((num_lines, *line_dims), dtype=leaf.dtype)

        decoding_state: DecodingState = jax.tree.map(empty_line_values, prefill_decoding_state)
        sampling_key, decoding_key = jax.random.split(key)
        return BlockContinuousState(
            decoding_state=decoding_state._replace(stop_flags=jnp.ones(num_lines, dtype=bool)),
            token_buffer=jnp.zeros((num_lines, max_output_length), dtype=jnp.int32),
            sampling_keys=jax.random.split(sampling_key, num_lines),
            decoding_keys=jax.random.split(decoding_key, num_lines),
            in_batch_index_to_sequence_id=jnp.zeros(num_lines, dtype=jnp.int32),
        )

    def empty_lines(self) -> list[int]:
        stopped = np.asarray(self.decoding_state.stop_flags)
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
        return bool(jnp.all(self.decoding_state.stop_flags))


type BlockStepCarry = tuple[
    DecodingState,
    Int[Array, "num_lines max_output"],
    Key[Array, " num_lines"],
    Key[Array, " num_lines"],
]


class BlockContinuousDecoder(eqx.Module):
    block_size: int
    max_output_length: int
    language_model: LanguageModel
    sampling_policy_template: SamplingPolicy
    decoding_batch_key: Key[Array, ""]
    speculator: Speculator

    @classmethod
    def init(
        cls,
        max_output_length: int,
        block_size: int,
        model: LanguageModel,
        sampling_policy: SamplingPolicy,
        decoding_batch_key: Key[Array, ""],
        speculator: Speculator,
    ) -> "BlockContinuousDecoder":
        return cls(
            block_size=block_size,
            max_output_length=max_output_length,
            language_model=model,
            sampling_policy_template=sampling_policy,
            decoding_batch_key=decoding_batch_key,
            speculator=speculator,
        )

    def initial_decoding_state(self, batch: PrefillBatch) -> DecodingState:
        return DecodingState.from_prefill(
            batch.prefill_results,
            batch.sampling_policy,
            self.speculator,
            jnp.int32,
        )

    def fill_lines(self, state: BlockContinuousState, batch: PrefillBatch) -> BlockContinuousState:
        prefill_decoding_state = self.initial_decoding_state(batch)

        empty_lines = state.decoding_state.stop_flags
        lines_passed = jnp.sum(batch.mask)
        source_idx = jnp.cumsum(empty_lines, dtype=jnp.int32) - 1
        need_updating = jnp.logical_and(empty_lines, source_idx < lines_passed)

        def apply_updates(
            new: Shaped[Array, "num_lines ..."],
            old: Shaped[Array, "num_lines ..."],
        ) -> Shaped[Array, "num_lines ..."]:
            return jax.vmap(jnp.where)(need_updating, new, old)

        gathered_prefill_state = jax.tree.map(lambda value: value[source_idx], prefill_decoding_state)
        new_decoding_state = jax.tree.map(apply_updates, gathered_prefill_state, state.decoding_state)

        return BlockContinuousState(
            decoding_state=new_decoding_state,
            token_buffer=apply_updates(jnp.zeros_like(state.token_buffer), state.token_buffer),
            sampling_keys=apply_updates(batch.sampling_keys[source_idx], state.sampling_keys),
            decoding_keys=apply_updates(batch.decoding_keys[source_idx], state.decoding_keys),
            in_batch_index_to_sequence_id=apply_updates(
                batch.sequence_ids[source_idx],
                state.in_batch_index_to_sequence_id,
            ),
        )

    def decode_block(self, state: BlockContinuousState) -> tuple[BlockContinuousState, Bool[Array, " num_lines"]]:
        eos_token_ids = jnp.asarray(self.language_model.config.generation_config.stop_token_ids, dtype=jnp.int32)
        decode_forward_pass_config = DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN)
        max_proposal_tokens = self.speculator.max_proposal_tokens
        (num_lines,) = state.in_batch_index_to_sequence_id.shape
        line_indices = jnp.arange(num_lines, dtype=jnp.int32)
        proposal_slots = jnp.arange(max_proposal_tokens, dtype=jnp.int32)

        def step(carry: BlockStepCarry, _: None) -> tuple[BlockStepCarry, None]:
            decoding_state, token_buffer, sampling_keys, decoding_keys = carry
            split_sampling_keys = jax.vmap(lambda key: jax.random.split(key, max_proposal_tokens + 1))(sampling_keys)
            next_sampling_keys = split_sampling_keys[:, 0]
            node_sampling_keys = split_sampling_keys[:, 1:]
            split_decoding_keys = jax.vmap(jax.random.split)(decoding_keys)
            next_decoding_keys = split_decoding_keys[:, 0]
            step_decoding_keys = split_decoding_keys[:, 1]

            new_decoding_state, step_results = self.language_model.decode_generation_step(
                decoding_state,
                (node_sampling_keys, step_decoding_keys),
                self.max_output_length,
                eos_token_ids,
                self.sampling_policy_template.has_count_penalties,
                None,
                decode_forward_pass_config,
                Keychain(
                    vmapped_keys=step_decoding_keys,
                    batch_key=self.decoding_batch_key,
                    sharding_config=self.language_model.sharding_config,
                ),
                self.speculator,
            )

            output_positions = decoding_state.num_generated_tokens[:, None] + proposal_slots[None, :]
            output_positions = jnp.where(
                proposal_slots[None, :] < step_results.lengths[:, None],
                output_positions,
                self.max_output_length,
            )
            new_token_buffer = token_buffer.at[line_indices[:, None], output_positions].set(
                step_results.token_ids,
                mode="drop",
            )
            return (new_decoding_state, new_token_buffer, next_sampling_keys, next_decoding_keys), None

        (new_decoding_state, new_token_buffer, new_sampling_keys, new_decoding_keys), _ = jax.lax.scan(
            step,
            (state.decoding_state, state.token_buffer, state.sampling_keys, state.decoding_keys),
            None,
            length=self.block_size,
        )
        completed_mask = jnp.logical_and(
            jnp.logical_not(state.decoding_state.stop_flags),
            new_decoding_state.stop_flags,
        )

        return BlockContinuousState(
            decoding_state=new_decoding_state,
            token_buffer=new_token_buffer,
            sampling_keys=new_sampling_keys,
            decoding_keys=new_decoding_keys,
            in_batch_index_to_sequence_id=state.in_batch_index_to_sequence_id,
        ), completed_mask


@dataclass(frozen=True)
class BatchScheduler(ABC):
    model: LanguageModel
    speculator: Speculator | None = None

    @abstractmethod
    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig | None = None,
        batch_scheduler_config: BatchSchedulerConfig = BatchSchedulerConfig(),
        *,
        fast_peak_memory: bool = False,
        keychain: Keychain | None = None,
    ) -> Iterator[tuple[int, GeneratedSequence]]: ...

    def reply_many(
        self,
        messages: Iterable[Iterable[Message]],
        generation_config: GenerationConfig | None = None,
        batch_scheduler_config: BatchSchedulerConfig = BatchSchedulerConfig(),
        *,
        enable_thinking: bool = True,
        keychain: Keychain | None = None,
        vram_bytes: int | None = None,
        batch_sizes_callback: Callable[[BatchSizesComputedEvent], None] | None = None,
    ) -> Iterator[tuple[int, AssistantMessage]]:
        messages = list(messages)
        keychain = (
            keychain or Keychain.init(0, shape=(len(messages),), sharding_config=self.model.sharding_config)
        ).broadcast((len(messages),))

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
                sharding_config=keychain.sharding_config,
            )

            for local_idx, result in self.generate_tokens_many(
                sequence_tokenized,
                generation_config=generation_config,
                batch_scheduler_config=bucket_config,
                fast_peak_memory=False,
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
        keychain: Keychain | None = None,
    ) -> Iterator[tuple[int, GeneratedSequence]]:
        tokenized = list(tokenized)
        if not tokenized:
            return
        if batch_scheduler_config.batch_size is None:
            raise RuntimeError("FixedSizeBatchScheduler requires batch_scheduler_config.batch_size.")
        batch_size = batch_scheduler_config.batch_size
        batch_axis = self.model.sharding_config.resolve_axis(LogicalAxis.BATCH)
        if batch_axis is None:
            batch_size_multiple = 1
        else:
            batch_axis_size = self.model.sharding_config.mesh.shape[batch_axis]
            if batch_size % batch_axis_size != 0:
                raise ValueError(
                    f"Batch size {batch_size} must be divisible by sharding axis {batch_axis!r} "
                    f"of size {batch_axis_size}.",
                )
            batch_size_multiple = batch_axis_size
        batch_token_sharding = self.model.sharding_config.make_sharding((batch_axis, None))
        batch_vector_sharding = self.model.sharding_config.make_sharding((batch_axis,))

        keychain = (
            keychain or Keychain.init(0, shape=(len(tokenized),), sharding_config=self.model.sharding_config)
        ).broadcast((len(tokenized),))

        def process_batches(current_batch_size: int) -> Iterator[tuple[int, GeneratedSequence]]:
            for batch_items in batched(enumerate(tokenized), current_batch_size):
                batch_with_ids = list(batch_items)
                batch_indices = [idx for idx, _ in batch_with_ids]
                real_batch = [tokens for _, tokens in batch_with_ids]
                padded_length = batch_scheduler_config.padded_length or max(len(tokens) for tokens in real_batch)
                num_padding_rows = current_batch_size - len(real_batch)
                lengths = jax.device_put(
                    jnp.asarray([len(tokens) for tokens in real_batch] + [1] * num_padding_rows, dtype=jnp.int32),
                    batch_vector_sharding,
                )
                padded = jax.device_put(
                    pad_sequences(real_batch, (current_batch_size, padded_length), dtype=jnp.int32),
                    batch_token_sharding,
                )
                padded_batch_indices = batch_indices + [0] * num_padding_rows
                padded_batch_indices_array = jnp.asarray(padded_batch_indices)
                batch_keychain = Keychain(
                    vmapped_keys=keychain.vmapped_keys.at[padded_batch_indices_array].get(
                        out_sharding=batch_vector_sharding,
                    ),
                    batch_key=keychain.batch_key,
                    sharding_config=keychain.sharding_config,
                ).squeeze()
                batch_results = self.model.generate_tokens(
                    padded,
                    generation_config=generation_config,
                    prompt_lengths_without_padding=lengths,
                    max_output_length=batch_scheduler_config.max_output_length,
                    num_top_logits_to_return=batch_scheduler_config.num_top_logits_to_return,
                    keychain=batch_keychain,
                    speculator=self.speculator,
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

        yield from _decrease_batchsize_on_oom(
            process_batches,
            starting_batch_size=batch_size,
            batch_size_multiple=batch_size_multiple,
        )


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
        keychain: Keychain | None = None,
    ) -> Iterator[tuple[int, GeneratedSequence]]:
        if batch_scheduler_config.num_top_logits_to_return is not None:
            raise RuntimeError("num_top_logits_to_return is not supported with ContinuousBatchScheduler.")

        tokenized = list(tokenized)
        if not tokenized:
            return
        if batch_scheduler_config.batch_size is None:
            raise RuntimeError("ContinuousBatchScheduler requires batch_scheduler_config.batch_size.")
        if batch_scheduler_config.padded_length is None:
            raise RuntimeError("ContinuousBatchScheduler requires batch_scheduler_config.padded_length.")

        keychain = (
            keychain or Keychain.init(0, shape=(len(tokenized),), sharding_config=self.model.sharding_config)
        ).broadcast((len(tokenized),))

        sampling_policy = (
            generation_config.default_policy() if generation_config else self.model.default_sampling_policy()
        )

        batch_size = batch_scheduler_config.batch_size
        max_output_length = batch_scheduler_config.max_output_length
        padded_length = batch_scheduler_config.padded_length
        if any(axis is not None for axis in self.model.sharding_config.logical_to_physical.values()):
            raise RuntimeError("ContinuousBatchScheduler does not support sharded models.")

        speculator = self.speculator
        if speculator is None:
            speculator = NoSpeculator.build(self.model.sharding_config)
        if not isinstance(speculator, NoSpeculator) and sampling_policy.has_count_penalties:
            raise ValueError("Speculator decoding only supports stateless sampling policies.")

        block_size = min(self.block_size, max_output_length)
        state_capacity = padded_length + max_output_length + speculator.max_proposal_tokens
        requested_prefill_batch_size = (
            batch_size
            if sampling_policy.has_count_penalties
            else max(1, min(int(batch_size * self.prefill_batch_fraction + 1), batch_size))
        )
        prefill_batch_size = requested_prefill_batch_size

        prefills: PrefillSource = PrefillSource(
            tokenized_messages=tokenized,
            keychain=keychain,
            prefill_batch_size=prefill_batch_size,
            padded_length=padded_length,
            chunk_size=self.prefill_chunk_size,
            state_capacity=state_capacity,
            sampling_policy_template=sampling_policy,
            speculator=speculator,
        )

        # make sure the key used is the same as the one in FixedBatchScheduler
        _, _, decoding_keychain = keychain.split(3)
        decoder = BlockContinuousDecoder.init(
            max_output_length=max_output_length,
            block_size=block_size,
            model=self.model,
            sampling_policy=sampling_policy,
            decoding_batch_key=decoding_keychain.batch_key,
            speculator=speculator,
        )

        jitted_decode = eqx.filter_jit(decoder.decode_block, donate="all")
        jitted_fill_lines = eqx.filter_jit(decoder.fill_lines, donate="all")

        prefills, partially_prefilled_batch = prefills.request(batch_size, decoder.language_model)
        state = BlockContinuousState.from_decoding_state(
            decoder.initial_decoding_state(partially_prefilled_batch),
            num_lines=batch_size,
            max_output_length=max_output_length,
            key=decoding_keychain.batch_key,
        )

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
