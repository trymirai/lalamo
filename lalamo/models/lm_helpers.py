import itertools
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from jax.errors import JaxRuntimeError

from lalamo.common import LalamoWarning, get_usable_memory_from_bytes
from lalamo.models.common import InferenceConfig

__all__ = [
    "BatchSizeEstimatingEvent",
    "decrease_batchsize_on_oom",
    "estimate_batchsize_from_bytes",
    "estimate_batchsizes_from_vram",
    "merge_small_buckets",
]


COMPILED_PROMPT_LENGTHS = [64 * 2**i for i in range(13)]


@dataclass(frozen=True)
class BatchSizeEstimatingEvent:
    lo: int
    hi: int | None


def merge_small_buckets(
    buckets: dict[int, list[tuple[int, np.ndarray]]],
    batch_size_for_length: dict[int, int],
    min_batches: int = 4,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    # Merge buckets that are too small into larger buckets.
    # Buckets smaller than min_batches * batch_size are merged into the next larger bucket.
    # The last bucket absorbs all overflow.
    sorted_lengths = sorted(buckets.keys())
    merged: dict[int, list[tuple[int, np.ndarray]]] = {}
    overflow: list[tuple[int, np.ndarray]] = []

    for i, padded_len in enumerate(sorted_lengths):
        batch_size = batch_size_for_length.get(padded_len, 1)  # note how with i's increment batch_size decreases
        items = overflow + buckets[padded_len]
        is_last = i == len(sorted_lengths) - 1

        if is_last:
            # add all the leftover items into the last bucket
            if items:
                merged[padded_len] = items
        elif len(items) < min_batches * batch_size:
            # the bucket is too small, push the items into a bigger one
            overflow = items
        else:
            # the bucket is big enough, keep rounded number of items and move on
            keep_count = (len(items) // batch_size) * batch_size
            merged[padded_len] = items[:keep_count]
            overflow = items[keep_count:]

    return merged


def estimate_batchsizes_from_vram(
    memory_consumption_callback: Callable[[InferenceConfig], int],
    sorted_lengths: list[int],
    vram_bytes: int,
    inference_config: InferenceConfig,
) -> dict[int, int]:
    # Estimate batch size for each bucket length via linear interpolation.
    # Since the map batch_size -> memory is convex, it can
    # be 'guaranteed' (as far as anything can be guaranteed in JAX) that this is a valid upper bound
    min_len, max_len = sorted_lengths[0], sorted_lengths[-1]
    assert min_len <= max_len

    def estimate_batchsize_for_seq_len(seq_len: int) -> int:
        def memory_per_batchsize(batch_size: int) -> int:
            config = InferenceConfig(
                max_output_length=inference_config.max_output_length,
                padded_length=seq_len,
                num_top_logits_to_return=inference_config.num_top_logits_to_return,
                batch_size=batch_size,
                sampling_policy=inference_config.sampling_policy,
            )
            return memory_consumption_callback(config)

        return estimate_batchsize_from_bytes(
            memory_per_batchsize,
            get_usable_memory_from_bytes(vram_bytes),
            progress=None,
        )

    if min_len == max_len:
        bs = estimate_batchsize_for_seq_len(min_len)
        return {min_len: max(1, bs)}

    bs_at_min = max(1, estimate_batchsize_for_seq_len(min_len))
    bs_at_max = max(1, estimate_batchsize_for_seq_len(max_len))

    def interpolate(length: int) -> int:
        t = (length - min_len) / (max_len - min_len)
        return max(1, int(bs_at_min + t * (bs_at_max - bs_at_min)))

    return {length: interpolate(length) for length in sorted_lengths}


def estimate_batchsize_from_bytes(
    memory_per_batchsize_callback: Callable[[int], int],
    target_mem_bytes: int,
    progress: Callable[[BatchSizeEstimatingEvent], None] | None = None,
) -> int:
    lo = 0
    hi = 0
    for candidate_exp in itertools.count():
        lo = hi
        hi = 2**candidate_exp

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
