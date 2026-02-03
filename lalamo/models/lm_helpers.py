import functools
import itertools
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import JaxRuntimeError

from lalamo.common import LalamoWarning, get_usable_memory_from_bytes
from lalamo.models.common import InferenceConfig

type TokenSequence = list[int] | np.ndarray | jnp.ndarray

__all__ = [
    "BatchSizeEstimatingEvent",
    "decrease_batchsize_on_oom",
    "estimate_batchsize_from_bytes",
    "estimate_batchsizes_from_vram",
    "merge_small_buckets",
    "pad_keys_to_size",
    "pad_sequences",
]


@dataclass(frozen=True)
class BatchSizeEstimatingEvent:
    lo: int
    hi: int | None


def _assert_sorted(values: list[int]) -> None:
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1)), "expected sorted inputs"


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
    pad_value: int = 0,
) -> jnp.ndarray:
    batch_size, seq_len = shape
    sequences_list = list(sequences)
    if len(sequences_list) > batch_size:
        raise ValueError(f"Expected at most {batch_size} sequences, got {len(sequences_list)}")

    if len(sequences_list) < batch_size:
        sequences_list.extend([np.array([pad_value], dtype=np.int32)] * (batch_size - len(sequences_list)))

    padded = np.full((batch_size, seq_len), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences_list):
        seq_arr = np.asarray(seq, dtype=np.int32)
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


def estimate_batchsizes_from_vram(
    memory_consumption_callback: Callable[[InferenceConfig], int],
    sorted_lengths: list[int],
    vram_bytes: int,
    inference_config: InferenceConfig,
) -> dict[int, int]:
    _assert_sorted(sorted_lengths)
    assert len(sorted_lengths) > 0
    usable_memory = get_usable_memory_from_bytes(vram_bytes)

    def memory_consumption(bs: int, seq_len: int) -> int:
        config = InferenceConfig(
            max_output_length=inference_config.max_output_length,
            padded_length=seq_len,
            num_top_logits_to_return=inference_config.num_top_logits_to_return,
            batch_size=bs,
        )
        return memory_consumption_callback(config)

    result: dict[int, int] = {}

    first_seq_len = sorted_lengths[0]
    bs = estimate_batchsize_from_bytes(functools.partial(memory_consumption, seq_len=sorted_lengths[0]), usable_memory)
    result[first_seq_len] = bs
    for seq_len in sorted_lengths[1:]:
        while bs > 1 and memory_consumption(bs, seq_len) > usable_memory:
            bs = max(1, int(bs * 0.8))

        result[seq_len] = bs

    return result


def estimate_batchsize_from_bytes(
    memory_per_batchsize_callback: Callable[[int], int],
    target_mem_bytes: int,
    progress: Callable[[BatchSizeEstimatingEvent], None] | None = None,
) -> int:
    lo = 0
    hi = 0
    for candidate_exp in itertools.count():
        lo = hi
        hi = 4**candidate_exp

        if progress is not None:
            progress(BatchSizeEstimatingEvent(lo, None))
        if target_mem_bytes < memory_per_batchsize_callback(hi):
            break

    while hi - lo > 1:
        mid = (lo + hi) // 2

        if progress is not None:
            progress(BatchSizeEstimatingEvent(lo, hi))
        if target_mem_bytes < memory_per_batchsize_callback(mid):
            hi = mid
        else:
            lo = mid

    return lo


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
        except JaxRuntimeError:
            if first_batch_completed:
                raise
            # because OOM's sometimes generate stuff that won't be garbage collected,
            # we need to be very aggressive with decreasing batchsize here
            new_bs = max(int(0.7 * effective_batch_size - 1), 1)
            if new_bs == 1 and effective_batch_size == 1:
                raise
            warnings.warn(
                f"OOM detected. Reducing batch size {effective_batch_size} -> {new_bs}.",
                LalamoWarning,
                stacklevel=3,
            )
            effective_batch_size = new_bs
