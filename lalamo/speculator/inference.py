from collections.abc import Callable, Iterable
from itertools import chain
from typing import NamedTuple

from lalamo.batching import BatchingConfig, FixedSizeBatching
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.data.utils import get_prefixes_ending_in_user_message
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import Message


class CollectTracesEvent(NamedTuple):
    sequences_processed: int
    tokens_generated: int


def inference_collect_traces(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    num_logits_per_token: int = 1024,
    trace_layers: tuple[int, ...] = (),
    batch_size: int = 1,
    max_input_length: int = 1024,
    max_output_length: int = 1024,
    tokens_to_generate: int | None = None,
    progress_callback: Callable[[CollectTracesEvent], None] | None = None,
) -> Iterable[LalamoCompletion]:
    if trace_layers:
        raise RuntimeError("Layer trace collection is not available on the current LanguageModel API.")

    prefixes = chain.from_iterable(map(get_prefixes_ending_in_user_message, conversations))
    tokenized_prefixes = map(model.token_codec.encode_request, prefixes)
    filtered_prefixes = filter(lambda conv: len(conv) <= max_input_length, tokenized_prefixes)
    filtered_prefixes = list(filtered_prefixes)  # eagerly materialize the prompts into RAM

    batching_config = BatchingConfig(
        max_output_length=max_output_length,
        padded_length=max_input_length,
        batch_size=batch_size,
        num_top_logits_to_return=num_logits_per_token,
    )
    tokens_generated = 0

    batching = FixedSizeBatching(model=model)
    for idx, generated in batching.generate_tokens_many(
        filtered_prefixes,
        generation_config=GenerationConfig(),
        batching_config=batching_config,
    ):
        token_ids = generated.token_ids.tolist()
        stop_token_ids = model.config.generation_config.stop_token_ids
        seqlen = next(
            (i + 1 for i, token_id in enumerate(token_ids) if token_id in stop_token_ids),
            len(token_ids),
        )

        if tokens_to_generate is not None:
            seqlen = min(seqlen, tokens_to_generate - tokens_generated)

        tokens_generated += seqlen
        token_ids = token_ids[:seqlen]

        if generated.top_k_token_ids is None or generated.top_k_token_logits is None:
            raise RuntimeError("Trace collection requires top-k token ids and logits.")

        yield LalamoCompletion(
            prefix_token_ids=filtered_prefixes[idx],
            completion_token_ids=token_ids,
            top_k_ids=generated.top_k_token_ids[:seqlen],
            top_k_logits=generated.top_k_token_logits[:seqlen],
        )

        if progress_callback is not None:
            progress_callback(CollectTracesEvent(idx + 1, tokens_generated))

        if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
            break
