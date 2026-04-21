import functools
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import JaxRuntimeError
from jaxtyping import DTypeLike

from lalamo.common import LalamoWarning

type TokenSequence = list[int] | np.ndarray | jnp.ndarray

PROMPT_SIZE_BUCKETS: tuple[int, ...] = tuple(256 * 4**i for i in range(8))

__all__ = [
    "PROMPT_SIZE_BUCKETS",
    "bucket_by_length",
    "bucket_sequences",
    "decrease_batchsize_on_oom",
    "estimate_batchsize_for_memory_budget",
    "merge_small_buckets",
    "pad_keys_to_size",
    "pad_sequences",
]


@dataclass(frozen=True)
class DeviceMemoryStats:
    bytes_in_use: int
    bytes_limit: int


def _get_device_memory_stats() -> DeviceMemoryStats:
    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None:
        raise RuntimeError(
            "Automatic batch size estimation is not supported on the current JAX backend, because"
            " device memory statistics are unavailable. This is expected to work on NVIDIA GPUs;"
            " if you are running on one, please open an issue. Otherwise, please specify"
            " --batch-size explicitly."
        )

    bytes_in_use = memory_stats.get("bytes_in_use")
    bytes_limit = memory_stats.get("bytes_limit")
    if bytes_in_use is None or bytes_limit is None:
        raise RuntimeError("Device memory_stats() must expose both bytes_in_use and bytes_limit.")

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


def estimate_batchsize_for_memory_budget(
    fn: Callable[[int], None],
    memory_budget: int,
    starting_batchsize: int = 2,
    num_steps: int = 2,
) -> int:
    batch_size = max(1, starting_batchsize)

    for _ in range(num_steps):
        try:
            peak_bytes_in_use = _measure_peak_bytes_in_use(functools.partial(fn, batch_size))
        except (JaxRuntimeError, ValueError) as error:
            if not _is_oom_error(error):
                raise

            # if we are out of memory, use a very conservative fallback
            warnings.warn(
                f"OOM while estimating batch size at {batch_size}. Sticking with {batch_size // 2}.",
                LalamoWarning,
                stacklevel=2,
            )

            return batch_size // 2

        batch_size = max(1, int(batch_size * memory_budget / peak_bytes_in_use))

    return batch_size


def bucket_by_length[T: TokenSequence](
    tokenized: list[T],
) -> dict[int, list[tuple[int, T]]]:
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
    min_batches_per_bucket: int = 10,
) -> tuple[dict[int, list[tuple[int, T]]], dict[int, int]]:
    """Bucket sequences by length and estimate a batch size per bucket.

    `memory_probe` is called as memory_probe(batch_size:int=..., padded_length:int=...) and
    must run at least one forward pass worth of work so the estimator can read the peak memory.
    """
    sequences_per_bucket = bucket_by_length(tokenized)
    sorted_lengths = sorted(sequences_per_bucket.keys(), reverse=True)
    batch_size_per_bucket: dict[int, int] = {}
    estimated = starting_batch_size
    had_successful_estimate = False
    num_steps = 2

    for i, padded_length in enumerate(sorted_lengths):
        try:
            estimated = estimate_batchsize_for_memory_budget(
                functools.partial(memory_probe, padded_length=padded_length),
                memory_budget=max_vram,
                starting_batchsize=estimated,
                num_steps=num_steps,
            )
        except (JaxRuntimeError, ValueError) as error:
            if not _is_oom_error(error):
                raise
            if had_successful_estimate:
                # Reuse the last good estimate for this bucket and every remaining one.
                batch_size_per_bucket.update(dict.fromkeys(sorted_lengths[i:], estimated))
                return sequences_per_bucket, batch_size_per_bucket
            # No prior estimate — fall back to 1 for this bucket and re-probe the next
            # bucket from scratch with num_steps=2.
            warnings.warn(
                f"Falling back to batch_size=1 for padded_length={padded_length}: {error}",
                LalamoWarning,
                stacklevel=2,
            )
            batch_size_per_bucket[padded_length] = 1
            estimated = 1
            num_steps = 2
            continue

        had_successful_estimate = True
        num_steps = 1
        batch_size_per_bucket[padded_length] = estimated

        remaining = sorted_lengths[i + 1 :]
        if sum(len(sequences_per_bucket[length]) for length in remaining) < min_batches_per_bucket * estimated:
            batch_size_per_bucket.update(dict.fromkeys(remaining, estimated))
            return sequences_per_bucket, batch_size_per_bucket

    return sequences_per_bucket, batch_size_per_bucket


def merge_small_buckets[T: TokenSequence](
    buckets: dict[int, list[tuple[int, T]]],
    batch_size_for_length: dict[int, int],
    min_batches: int = 4,
) -> dict[int, list[tuple[int, T]]]:
    # Merge buckets that are too small into larger buckets.
    # Buckets smaller than min_batches * batch_size are merged into the next larger bucket.
    # The last bucket absorbs all overflow.
    sorted_lengths = sorted(buckets.keys())
    merged: dict[int, list[tuple[int, T]]] = {}
    overflow: list[tuple[int, T]] = []

    for padded_len in sorted_lengths:
        batch_size = batch_size_for_length.get(padded_len, 1)  # note how with i's increment batch_size decreases
        items = overflow + buckets[padded_len]

        if len(items) < min_batches * batch_size:
            # the bucket is too small, push the items into a bigger one
            overflow = items
        else:
            # the bucket is big enough, keep _all_ the items and move on
            # keeping all the items avoids a funny problem with spill over into _very_ long ctx length
            merged[padded_len] = items
            overflow = []

    if overflow:
        # any leftover items go into the largest bucket
        largest_len = sorted_lengths[-1]
        merged.setdefault(largest_len, []).extend(overflow)

    return merged


def pad_sequences(
    sequences: Iterable[TokenSequence],
    shape: tuple[int, int],
    *,
    dtype: DTypeLike,
    pad_value: int = 0,
) -> jnp.ndarray:
    batch_size, seq_len = shape
    sequences_list = list(sequences)
    if len(sequences_list) > batch_size:
        raise ValueError(f"Expected at most {batch_size} sequences, got {len(sequences_list)}")

    if len(sequences_list) < batch_size:
        sequences_list.extend([jnp.array([pad_value], dtype=dtype)] * (batch_size - len(sequences_list)))

    padded = np.full((batch_size, seq_len), pad_value, dtype=dtype)
    for i, seq in enumerate(sequences_list):
        seq_arr = np.asarray(seq, dtype=dtype)
        if seq_arr.size > seq_len:
            raise ValueError(f"Sequence length {seq_arr.size} exceeds target length {seq_len}")
        padded[i, : seq_arr.size] = seq_arr

    return jnp.array(padded)


def pad_keys_to_size(keys: Iterable, size: int, *, seed: int = 0) -> jnp.ndarray:
    keys_list = list(keys)
    if len(keys_list) > size:
        raise ValueError(f"Expected at most {size} keys, got {len(keys_list)}")
    if len(keys_list) == size:
        return jnp.array(keys_list)
    dummy_keys = jax.random.split(jax.random.key(seed), size - len(keys_list))
    return jnp.concatenate([jnp.array(keys_list), dummy_keys])


def _is_oom_error(e: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()


def decrease_batchsize_on_oom[T](
    fn: Callable[[int], Iterable[T]],
    starting_batch_size: int,
) -> Iterable[T]:
    first_batch_completed = False
    effective_batch_size = starting_batch_size

    while True:
        try:
            for result in fn(effective_batch_size):
                yield result

                # as soon as we yielded we are not allowed to retry anymore
                # to make sure we don't ever miss/duplicate outputs
                first_batch_completed = True
            break
        except (JaxRuntimeError, ValueError) as e:
            if not _is_oom_error(e) or first_batch_completed:
                raise
            # because OOM's sometimes generate stuff that won't be garbage collected,
            # we need to be very aggressive with decreasing batchsize here
            new_bs = max(int(0.7 * effective_batch_size - 1), 1)
            if new_bs == 1 and effective_batch_size == 1:
                raise

            warnings.warn(
                f"OOM detected. Reducing batch size {effective_batch_size} -> {new_bs}.\n"
                f"Full traceback:\n{traceback.format_exc()}",
                LalamoWarning,
                stacklevel=3,
            )
            effective_batch_size = new_bs
