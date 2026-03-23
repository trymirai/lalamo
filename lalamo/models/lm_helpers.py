import json
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

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


def _predict_max_batch_sizes(
    probes: list[MemoryProbe],
    measured: list[int],
    sorted_lengths: list[int],
    memory_budget: int,
) -> dict[int, int]:
    """Fit mem = a + b*bs + c*seq_len + d*bs*seq_len and solve for max bs per length."""
    matrix = np.array(
        [[1, p.batch_size, p.seq_len, p.batch_size * p.seq_len] for p in probes],
        dtype=np.float64,
    )
    target = np.array(measured, dtype=np.float64)
    (a, b, c, d), *_ = np.linalg.lstsq(matrix, target, rcond=None)

    result: dict[int, int] = {}
    for sl in sorted_lengths:
        denominator = b + d * sl
        if denominator <= 0:
            result[sl] = 1
        else:
            result[sl] = max(1, int((memory_budget - a - c * sl) / denominator))
    return result


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


def _run_probes_sequential(
    model_path: Path | str,
    probes: list[MemoryProbe],
    max_output_length: int,
    num_logits_to_return: int | None,
    progress: Callable[[BatchSizeEstimatingEvent], None] | None = None,
) -> list[int]:
    from lalamo.models import LanguageModelConfig
    from lalamo.models.common import InferenceConfig

    model = LanguageModelConfig.load_model(model_path)
    measured: list[int] = []
    for i, probe in enumerate(probes):
        if progress is not None:
            progress(BatchSizeEstimatingEvent(i, len(probes)))
        config = InferenceConfig(
            max_output_length=max_output_length,
            padded_length=probe.seq_len,
            num_top_logits_to_return=num_logits_to_return,
            batch_size=probe.batch_size,
        )
        measured.append(model.estimate_memory_consumption(inference_config=config))
    return measured


def _run_probes_parallel(
    model_path: Path | str,
    probes: list[MemoryProbe],
    max_output_length: int,
    num_logits_to_return: int | None,
    progress: Callable[[BatchSizeEstimatingEvent], None] | None = None,
    max_concurrent: int = 4,
) -> list[int]:
    """Run memory probes in parallel GPU subprocesses (waves of max_concurrent).

    Falls back to sequential in-process compilation if any subprocess fails with OOM.
    """
    measured: list[int] = []
    probe_idx = 0
    num_logits_arg = str(num_logits_to_return) if num_logits_to_return is not None else "none"

    for wave_start in range(0, len(probes), max_concurrent):
        wave = probes[wave_start : wave_start + max_concurrent]
        procs: list[subprocess.Popen[bytes]] = []
        for probe in wave:
            if progress is not None:
                progress(BatchSizeEstimatingEvent(probe_idx, len(probes)))
            procs.append(subprocess.Popen(
                [
                    sys.executable, "-m", "lalamo.models._memory_probe_worker",
                    str(model_path), str(probe.batch_size), str(probe.seq_len),
                    str(max_output_length), num_logits_arg,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ))
            probe_idx += 1

        for proc in procs:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                stderr_text = stderr.decode()
                if "resource_exhausted" in stderr_text.lower() or "out of memory" in stderr_text.lower():
                    for p in procs:
                        p.kill()
                        p.wait()
                    warnings.warn(
                        "Parallel memory probe failed (GPU OOM in subprocess). "
                        "Falling back to sequential in-process compilation.",
                        LalamoWarning,
                        stacklevel=2,
                    )
                    return _run_probes_sequential(
                        model_path, probes, max_output_length, num_logits_to_return, progress,
                    )
                raise RuntimeError(f"Memory probe worker failed (exit {proc.returncode}): {stderr_text[-500:]}")
            measured.append(json.loads(stdout.decode())["memory_bytes"])

    return measured


def estimate_batchsizes_from_vram(
    model_path: Path | str,
    sorted_lengths: list[int],
    vram_bytes: int,
    inference_config: InferenceConfig,
    progress: Callable[[BatchSizeEstimatingEvent], None] | None = None,
) -> dict[int, int]:
    _assert_sorted(sorted_lengths)
    assert len(sorted_lengths) > 0
    usable_memory = get_usable_memory_from_bytes(vram_bytes)

    len_short = max(sorted_lengths[0], 512)
    len_long = max(sorted_lengths[-1], 512)
    probe_lengths = [len_short] if len_short == len_long else [len_short, len_long]

    max_out = inference_config.max_output_length
    num_logits = inference_config.num_top_logits_to_return
    bs_low = 4

    # Wave 1: probe at small batch size to measure per-sample cost
    wave1_probes = [MemoryProbe(bs_low, sl) for sl in probe_lengths]
    wave1_measured = _run_probes_parallel(
        model_path, wave1_probes, max_out, num_logits, progress,
    )

    # Estimate bs_high per seq_len from wave 1
    bs_high_per_len: dict[int, int] = {}
    for probe, mem in zip(wave1_probes, wave1_measured):
        per_sample = mem / bs_low
        bs_high_per_len[probe.seq_len] = (
            max(bs_low * 2, int(usable_memory * 0.8 / per_sample)) if per_sample > 0 else bs_low * 2
        )

    # Wave 2: probe near the estimated operating point
    wave2_probes = [MemoryProbe(bs_high_per_len[sl], sl) for sl in probe_lengths]
    wave2_measured = _run_probes_parallel(
        model_path, wave2_probes, max_out, num_logits, progress,
    )

    return _predict_max_batch_sizes(
        wave1_probes + wave2_probes,
        wave1_measured + wave2_measured,
        sorted_lengths,
        usable_memory,
    )


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
        except JaxRuntimeError as e:
            if first_batch_completed:
                raise
            # because OOM's sometimes generate stuff that won't be garbage collected,
            # we need to be very aggressive with decreasing batchsize here
            new_bs = max(int(0.7 * effective_batch_size - 1), 1)
            if new_bs == 1 and effective_batch_size == 1:
                raise
            warnings.warn(
                f"OOM detected ({e}). Reducing batch size {effective_batch_size} -> {new_bs}.",
                LalamoWarning,
                stacklevel=3,
            )
            effective_batch_size = new_bs
