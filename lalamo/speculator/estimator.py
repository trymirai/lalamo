import functools
import itertools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lalamo.models import LanguageModel


def get_default_device_memory() -> int | None:
    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None or "bytes_limit" not in memory_stats:
        return None
    return memory_stats["bytes_limit"]


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
            backend="cpu",  # cuda backend tries to allocate in .compile() and ooms
        )
        .lower(
            model,
            prompt_token_ids=jax.ShapeDtypeStruct((batch_size, max_input_length), jnp.int32),
            prompt_lengths_without_padding=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        )
        .compile()
        .memory_analysis()
    )

    assert hasattr(memory_analysis, "argument_size_in_bytes")
    assert hasattr(memory_analysis, "output_size_in_bytes")
    assert hasattr(memory_analysis, "temp_size_in_bytes")

    return (
        memory_analysis.argument_size_in_bytes  # type: ignore (pyright bug)
        + memory_analysis.output_size_in_bytes  # type: ignore (pyright bug)
        + memory_analysis.temp_size_in_bytes  # type: ignore (pyright bug)
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
