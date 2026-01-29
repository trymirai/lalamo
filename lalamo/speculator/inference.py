from collections.abc import Callable, Iterable
from itertools import chain
from typing import NamedTuple

import jax

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.data.utils import get_prefixes_ending_in_user_message
from lalamo.message_processor import Message
from lalamo.models import LanguageModel


class CollectTracesEvent(NamedTuple):
    sequences_processed: int
    tokens_generated: int


def inference_collect_traces(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    num_top_logits_to_collect: int = 8,
    batch_size: int = 1,
    max_input_length: int = 1024,
    max_output_length: int = 1024,
    tokens_to_generate: int | None = None,
    progress_callback: Callable[[CollectTracesEvent], None] | None = None,
) -> Iterable[LalamoCompletion]:
    prefixes = chain.from_iterable(map(get_prefixes_ending_in_user_message, conversations))
    tokenized_prefixes = map(model.message_processor.tokenize_request, prefixes)

    tokens_generated, sequences_processed = 0, 0
    batch_count = 0

    for result in model.generate_tokens_batched_safe(
        tokenized_prefixes,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        num_top_logits_to_return=num_top_logits_to_collect,
        key=jax.random.PRNGKey(0),
    ):
        # Truncate based on tokens_to_generate limit
        output_ids = result.output_token_ids
        if tokens_to_generate is not None:
            remaining = tokens_to_generate - tokens_generated
            output_ids = output_ids[:remaining]

        tokens_generated += len(output_ids)
        sequences_processed += 1
        batch_count += 1

        # Build token logits dict
        token_logits = []
        if result.top_k_token_ids is not None and result.top_k_token_logits is not None:
            for keys, values in zip(
                result.top_k_token_ids[: len(output_ids)],
                result.top_k_token_logits[: len(output_ids)],
                strict=True,
            ):
                token_logits.append(dict(zip(keys, values, strict=True)))

        yield LalamoCompletion(result.input_token_ids, output_ids, token_logits)

        # Progress callback after each batch worth of results
        if batch_count >= batch_size:
            if progress_callback is not None:
                progress_callback(CollectTracesEvent(sequences_processed, tokens_generated))
            batch_count = 0

        if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
            break

    # Final progress callback for any remaining
    if progress_callback is not None and batch_count > 0:
        progress_callback(CollectTracesEvent(sequences_processed, tokens_generated))
