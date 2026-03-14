import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import JaxRuntimeError
from jaxtyping import DTypeLike

from lalamo.common import LalamoWarning, get_usable_memory_from_bytes
from lalamo.models.common import InferenceConfig

type TokenSequence = list[int] | np.ndarray | jnp.ndarray

__all__ = [
    "BatchSizeEstimatingEvent",
    "MemoryPredictor",
    "MemoryProbe",
    "decrease_batchsize_on_oom",
    "estimate_batchsizes_from_vram",
    "merge_small_buckets",
    "pad_keys_to_size",
    "pad_sequences",
]


@dataclass(frozen=True)
class BatchSizeEstimatingEvent:
    num_compilations_done: int
    num_compilations_total: int


@dataclass(frozen=True)
class MemoryProbe:
    batch_size: int
    seq_len: int


class MemoryPredictor:
    """Fits mem = a + b·bs + c·seq_len + d·bs·seq_len from probe measurements."""

    def __init__(self) -> None:
        self._coefficients: np.ndarray | None = None

    def fit(self, probes: list[MemoryProbe], measured: list[int]) -> None:
        matrix = np.array(
            [[1, p.batch_size, p.seq_len, p.batch_size * p.seq_len] for p in probes],
            dtype=np.float64,
        )
        target = np.array(measured, dtype=np.float64)
        self._coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)

    def predict(self, batch_size: int, seq_len: int) -> int:
        assert self._coefficients is not None
        a, b, c, d = self._coefficients
        return int(a + b * batch_size + c * seq_len + d * batch_size * seq_len)

    def max_batch_size(self, seq_len: int, memory_budget: int, *, safety_margin: float = 0.9) -> int:
        assert self._coefficients is not None
        a, b, c, d = self._coefficients
        numerator = memory_budget * safety_margin - a - c * seq_len
        denominator = b + d * seq_len
        if denominator <= 0:
            return 1
        return max(1, int(numerator / denominator))


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


_PROBE_BATCH_SIZES = (8, 32)


def estimate_batchsizes_from_vram(
    memory_consumption_callback: Callable[[InferenceConfig], int],
    sorted_lengths: list[int],
    vram_bytes: int,
    inference_config: InferenceConfig,
    progress: Callable[[BatchSizeEstimatingEvent], None] | None = None,
) -> dict[int, int]:
    _assert_sorted(sorted_lengths)
    assert len(sorted_lengths) > 0
    usable_memory = get_usable_memory_from_bytes(vram_bytes)

    len_short, len_long = sorted_lengths[0], sorted_lengths[-1]
    probe_lengths = (len_short,) if len_short == len_long else (len_short, len_long)
    probes = [MemoryProbe(bs, sl) for bs in _PROBE_BATCH_SIZES for sl in probe_lengths]

    measured: list[int] = []
    for i, probe in enumerate(probes):
        if progress is not None:
            progress(BatchSizeEstimatingEvent(i, len(probes)))
        config = InferenceConfig(
            max_output_length=inference_config.max_output_length,
            padded_length=probe.seq_len,
            num_top_logits_to_return=inference_config.num_top_logits_to_return,
            batch_size=probe.batch_size,
        )
        measured.append(memory_consumption_callback(config))

    predictor = MemoryPredictor()
    predictor.fit(probes, measured)
    return {sl: predictor.max_batch_size(sl, usable_memory) for sl in sorted_lengths}


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
