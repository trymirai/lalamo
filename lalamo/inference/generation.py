from __future__ import annotations

import functools
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from lalamo.common import decrease_batchsize_on_oom
from lalamo.inference.estimator import EstimateBatchsizeFromMemoryEvent, estimate_batchsize_from_memory
from lalamo.models import LanguageModel
from lalamo.models.language_model import GenerationResults

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from jax._src.stages import Compiled
    from jaxtyping import PRNGKeyArray

    from lalamo.message_processor import AssistantMessage, Message
    from lalamo.models.language_model import ForwardPassConfig
    from lalamo.sampling import SamplingPolicy


__all__ = [
    "BatchSizeEstimatingEvent",
    "GenerateConfig",
    "generate_batched",
    "reply_many",
]


COMPILED_PROMPT_LENGTHS = [64 * 2**i for i in range(13)]
MIN_BATCHES_IN_BUCKET = 5


@dataclass(frozen=True)
class BatchSizeEstimatingEvent:
    sequence_length: int
    lo: int
    hi: int | None


@dataclass(frozen=True)
class GenerateConfig:
    sampling_policy: SamplingPolicy | None = None
    forward_pass_config: ForwardPassConfig | None = None
    max_output_length: int = 8192
    num_top_logits_to_return: int | None = None
    key: PRNGKeyArray | None = None


def _bucket_by_length(
    tokenized: list[np.ndarray],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    # Group tokenized sequences by their padded length.
    # Returns a dict mapping padded_length -> list of (original_index, tokens).
    bucket_lengths = COMPILED_PROMPT_LENGTHS

    buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
    for idx, tokens in enumerate(tokenized):
        padded_len = min(length for length in bucket_lengths if length >= len(tokens))
        buckets.setdefault(padded_len, []).append((idx, tokens))
    return buckets


def _merge_small_buckets(
    buckets: dict[int, list[tuple[int, np.ndarray]]],
    batch_size_for_length: dict[int, int],
    min_batches: int = MIN_BATCHES_IN_BUCKET,
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


def _linear_estimate_batch_sizes(
    model: LanguageModel,
    sorted_lengths: list[int],
    max_output_length: int,
    max_vram: int,
    progress_callback: Callable[[BatchSizeEstimatingEvent], None] | None = None,
) -> dict[int, int]:
    # Estimate batch size for each bucket length via linear interpolation.
    # Since the map batch_size -> memory is convex, it can
    # be 'guaranteed' (as far as anything can be guaranteed in JAX) that this is a valid upper bound
    min_len, max_len = sorted_lengths[0], sorted_lengths[-1]

    def make_progress(seq_len: int) -> Callable[[EstimateBatchsizeFromMemoryEvent], None] | None:
        if progress_callback is None:
            return None

        def inner(event: EstimateBatchsizeFromMemoryEvent) -> None:
            progress_callback(BatchSizeEstimatingEvent(seq_len, event.lo, event.hi))

        return inner

    if min_len == max_len:
        bs = estimate_batchsize_from_memory(model, min_len, max_output_length, None, max_vram, make_progress(min_len))
        return {min_len: max(1, bs)}

    bs_at_min = max(
        1,
        estimate_batchsize_from_memory(model, min_len, max_output_length, None, max_vram, make_progress(min_len)),
    )
    bs_at_max = max(
        1,
        estimate_batchsize_from_memory(model, max_len, max_output_length, None, max_vram, make_progress(max_len)),
    )

    def interpolate(length: int) -> int:
        t = (length - min_len) / (max_len - min_len)
        return max(1, int(bs_at_min + t * (bs_at_max - bs_at_min)))

    return {length: interpolate(length) for length in sorted_lengths}


def generate_batched(
    model: LanguageModel,
    tokenized: Iterable[tuple[int, np.ndarray]],
    padded_length: int,
    batch_size: int,
    config: GenerateConfig | None = None,
) -> Iterator[tuple[int, GenerationResults]]:
    # Generate for a bunch of tokenized inputs with batch-size decrease on OOM

    tokenized_list = list(tokenized)

    @functools.cache
    def make_compiled(bs: int) -> Compiled:
        return (
            jax.jit(
                functools.partial(
                    LanguageModel.generate_tokens,
                    sampling_policy=config.sampling_policy,
                    max_output_length=config.max_output_length,
                    forward_pass_config=config.forward_pass_config,
                    num_top_logits_to_return=config.num_top_logits_to_return,
                    key=config.key,
                ),
            )
            .lower(
                model,
                prompt_token_ids=jax.ShapeDtypeStruct((bs, padded_length), jnp.int32),
                prompt_lengths_without_padding=jax.ShapeDtypeStruct((bs,), jnp.int32),
            )
            .compile(compiler_options={"xla_gpu_autotune_level": "0"})
        )

    def process_batches(bs: int) -> Iterator[tuple[int, GenerationResults]]:
        for real_batch in batched(tokenized_list, bs):
            indices = [idx for idx, _ in real_batch]
            tokens_list = [tokens for _, tokens in real_batch]

            batch_padding = bs - len(tokens_list)
            batch = (*tokens_list, *(np.array([0], dtype=np.int32),) * batch_padding)

            padded = jnp.array(
                np.array([np.pad(tokens, (0, padded_length - len(tokens)), constant_values=0) for tokens in batch]),
            )
            lengths = jnp.array([len(tokens) for tokens in batch], dtype=jnp.int32)

            compiled = make_compiled(bs)
            results = compiled(
                model,
                prompt_token_ids=padded,
                prompt_lengths_without_padding=lengths,
            )

            for i, idx in enumerate(indices):
                yield (
                    idx,
                    GenerationResults(
                        token_ids=results.token_ids[i],
                        top_k_token_ids=results.top_k_token_ids[i] if results.top_k_token_ids is not None else None,
                        top_k_token_logits=results.top_k_token_logits[i]
                        if results.top_k_token_logits is not None
                        else None,
                    ),
                )

    yield from decrease_batchsize_on_oom(process_batches, batch_size)


def reply_many(
    model: LanguageModel,
    messages: Iterable[list[Message]],
    max_vram: int | None,
    config: GenerateConfig = GenerateConfig(),  # noqa: B008
    batch_size: int | None = None,
    estimating_progress_callback: Callable[[BatchSizeEstimatingEvent], None] | None = None,
    estimated_callback: Callable[[], None] | None = None,
) -> Iterator[tuple[int, AssistantMessage]]:
    # Generate replies for multiple message sequences with automatic batching.

    # Sequences are grouped by padded input length and processed longest-first,
    # so output order may differ from input order. Each result includes its original index.

    # check that only one is not None
    assert (batch_size is None) == (max_vram is not None)

    tokenized = [
        np.array(
            model.message_processor.tokenize_text(model.message_processor.render_request(msg)),
            dtype=np.int32,
        )
        for msg in messages
    ]

    buckets = _bucket_by_length(tokenized)
    sorted_lengths = sorted(buckets.keys())

    if batch_size is not None:
        # Use fixed batch size for all lengths, skip estimation
        batch_size_for_length = dict.fromkeys(sorted_lengths, batch_size)
    else:
        batch_size_for_length = _linear_estimate_batch_sizes(
            model,
            sorted_lengths,
            config.max_output_length,
            max_vram,  # ty: ignore
            estimating_progress_callback,
        )
        buckets = _merge_small_buckets(buckets, batch_size_for_length)

    if estimated_callback is not None:
        estimated_callback()

    # Process longest sequences first so batchsize=1 OOM happens as early as possible, if it does happen
    for padded_length in sorted(buckets.keys(), reverse=True):
        bucket = buckets[padded_length]
        batch_size = batch_size_for_length.get(padded_length, 1)

        for idx, results in generate_batched(model, bucket, padded_length, batch_size, config):
            response_text = model.message_processor.detokenize(results.token_ids.tolist())
            yield idx, model.message_processor.parse_response(response_text)
