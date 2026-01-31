from collections.abc import Callable, Iterable
from itertools import chain
from typing import NamedTuple

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.data.utils import get_prefixes_ending_in_user_message
from lalamo.message_processor import Message
from lalamo.models import LanguageModel
from lalamo.models.common import InferenceConfig


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
    filtered_prefixes = filter(lambda conv: len(conv) <= max_input_length, tokenized_prefixes)
    filtered_prefixes = list(filtered_prefixes)  # eagerly materialize the prompts into RAM

    config = InferenceConfig(
        max_output_length=max_output_length,
        num_top_logits_to_return=num_top_logits_to_collect,
        padded_length=max_input_length,
        batch_size=batch_size,
    )

    tokens_generated = 0

    for idx, generated in enumerate(
        model.generate_tokens_many(
            filtered_prefixes,
            inference_config=config,
        ),
    ):
        token_ids = generated.token_ids.tolist()
        seqlen = next(
            (i + 1 for i, t in enumerate(token_ids) if t in model.stop_token_ids),
            len(token_ids),
        )

        if tokens_to_generate is not None:
            seqlen = min(seqlen, tokens_to_generate - tokens_generated)

        tokens_generated += seqlen

        token_ids = token_ids[:seqlen]

        assert generated.top_k_token_ids is not None and generated.top_k_token_logits is not None
        token_logits_ids = generated.top_k_token_ids[:seqlen].tolist()
        token_logits_values = generated.top_k_token_logits[:seqlen].tolist()
        token_logits = [
            dict(zip(keys, values, strict=True))
            for keys, values in zip(token_logits_ids, token_logits_values, strict=True)
        ]

        # We need the original prompt tokens - get from indexed_inputs
        yield LalamoCompletion(filtered_prefixes[idx], token_ids, token_logits)

        if progress_callback is not None:
            progress_callback(CollectTracesEvent(idx, tokens_generated))

        if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
            break
