import functools
import itertools
import os
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lalamo.models import LanguageModel


def get_default_device_bytes() -> int | None:
    dynamic_allocate = False

    preallocate = os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE", "")
    dynamic_allocate |= preallocate.strip().lower() in {"0", "false", "no", "off"}

    allocator = os.getenv("XLA_PYTHON_CLIENT_ALLOCATOR", "")
    dynamic_allocate |= allocator.strip().lower() in {"platform", "cuda_malloc_async"}

    if dynamic_allocate:
        return None

    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None or "bytes_limit" not in memory_stats:
        return None

    mem_fraction_raw = os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "")
    try:
        mem_fraction = float(mem_fraction_raw)
    except ValueError:
        mem_fraction = 0.75  # jax default https://docs.jax.dev/en/latest/gpu_memory_allocation.html

    # 500mb is seemingly the usually observed overhead; this tries to match the actual capacity of the gpu
    # so it should correspond to something you'd see in nvidia-smi
    memory_limit = memory_stats["bytes_limit"] / min(mem_fraction, 1.0) + (500 * 1000 * 1000)

    return get_usable_memory_from_bytes(memory_limit)


def get_usable_memory_from_bytes(limit_bytes: int) -> int:
    # JAX allocates a bit more than it needs, so we discount it by some safety factor
    return int(limit_bytes * 0.93)


def estimate_memory_from_batchsize(
    model: LanguageModel,
    max_input_length: int,
    max_output_length: int,
    num_logits_per_token: int,
    batch_size: int,
) -> int:
    memory_analysis = (
        jax.jit(
            functools.partial(
                LanguageModel.generate_tokens,
                max_output_length=max_output_length,
                num_top_logits_to_return=num_logits_per_token,
            ),
        )
        .lower(
            model,
            prompt_token_ids=jax.ShapeDtypeStruct((batch_size, max_input_length), jnp.int32),
            prompt_lengths_without_padding=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        )
        # disables autotune, see https://guides.lw1.at/all-xla-options/#--xla_gpu_autotune_level
        .compile(compiler_options={"xla_gpu_autotune_level": "0"})
        .memory_analysis()
    )

    assert hasattr(memory_analysis, "argument_size_in_bytes")
    assert hasattr(memory_analysis, "output_size_in_bytes")
    assert hasattr(memory_analysis, "temp_size_in_bytes")

    return (
        memory_analysis.argument_size_in_bytes
        + memory_analysis.output_size_in_bytes
        + memory_analysis.temp_size_in_bytes
    )


class EstimateBatchsizeFromMemoryEvent(NamedTuple):
    lo: int
    hi: int | None


def estimate_batchsize_from_memory(
    model: LanguageModel,
    max_input_length: int,
    max_output_length: int,
    num_logits_per_token: int,
    target_mem: int,
    progress: Callable[[EstimateBatchsizeFromMemoryEvent], None] | None = None,
) -> int:
    mem_for_bs = functools.cache(
        functools.partial(
            estimate_memory_from_batchsize,
            model,
            max_input_length,
            max_output_length,
            num_logits_per_token,
        ),
    )

    lo = 0
    hi = 0
    for candidate_exp in itertools.count():
        lo = hi
        hi = 2**candidate_exp

        if progress is not None:
            progress(EstimateBatchsizeFromMemoryEvent(lo, None))
        if target_mem < mem_for_bs(hi):
            break

    while hi - lo > 1:
        mid = (lo + hi) // 2

        if progress is not None:
            progress(EstimateBatchsizeFromMemoryEvent(lo, hi))
        if target_mem < mem_for_bs(mid):
            hi = mid
        else:
            lo = mid

    return lo
