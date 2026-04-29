from collections.abc import Callable, Iterable
from itertools import chain
from typing import NamedTuple

import numpy as np

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.data.utils import get_prefixes_ending_in_user_message
from lalamo.message_processor import Message
from lalamo.models import GenerationConfig, GenerationTraceConfig, LanguageModel
from lalamo.models.batch_scheduler import FixedBatchScheduler
from lalamo.models.common import InferenceConfig
from lalamo.modules import MLPForwardPassConfig


class CollectTracesEvent(NamedTuple):
    sequences_processed: int
    tokens_generated: int


def inference_collect_traces(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    num_logits_per_token: int = 1024,
    trace_layers: tuple[int, ...] = tuple(),
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
        padded_length=max_input_length,
        batch_size=batch_size,
    )
    trace_config = GenerationTraceConfig(
        num_logits_per_token=num_logits_per_token,
        trace_layers=trace_layers,
    )

    tokens_generated = 0

    scheduler = FixedBatchScheduler(model=model)

    for idx, generated in scheduler.generate_tokens_many(
        filtered_prefixes,
        generation_config=GenerationConfig(),
        inference_config=config,
        forward_pass_config=MLPForwardPassConfig(),
        generation_trace_config=trace_config,
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

        assert generated.trace is not None
        trace = generated.trace
        layer_indices = tuple(np.asarray(trace.layer_indices, dtype=np.int32).tolist())

        yield LalamoCompletion(
            prefix_token_ids=filtered_prefixes[idx],
            completion_token_ids=token_ids,
            top_k_ids=trace.top_k_ids[:seqlen],
            top_k_logits=trace.top_k_logits[:seqlen],
            logsumexp=trace.logsumexp[:seqlen],
            activation_output=trace.activation_output[:seqlen],
            layer_indices=layer_indices,
            layer_output=trace.layer_output[:, :seqlen, :] if layer_indices else None,
        )

        if progress_callback is not None:
            progress_callback(CollectTracesEvent(idx + 1, tokens_generated))

        if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
            break
