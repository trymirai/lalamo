from collections.abc import Callable, Iterable
from itertools import chain
from typing import NamedTuple

from einops import rearrange

from lalamo.data.trace_shard import TraceCompletionRecord
from lalamo.data.utils import get_prefixes_ending_in_user_message
from lalamo.message_processor import Message
from lalamo.models import GenerationTraceConfig, LanguageModel
from lalamo.models.common import InferenceConfig


class CollectTracesEvent(NamedTuple):
    sequences_processed: int
    tokens_generated: int


def inference_collect_traces(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    raw_top_k: int = 8192,
    trace_layers: tuple[int, ...] = tuple(),
    batch_size: int = 1,
    max_input_length: int = 1024,
    max_output_length: int = 1024,
    tokens_to_generate: int | None = None,
    progress_callback: Callable[[CollectTracesEvent], None] | None = None,
) -> Iterable[TraceCompletionRecord]:
    prefixes = chain.from_iterable(map(get_prefixes_ending_in_user_message, conversations))
    tokenized_prefixes = map(model.message_processor.tokenize_request, prefixes)
    filtered_prefixes = filter(lambda conv: len(conv) <= max_input_length, tokenized_prefixes)
    filtered_prefixes = list(filtered_prefixes)  # eagerly materialize the prompts into RAM

    config = InferenceConfig(
        max_output_length=max_output_length,
        padded_length=max_input_length,
        batch_size=batch_size,
    )
    generation_trace_config = GenerationTraceConfig(
        raw_top_k=raw_top_k,
        return_output_norm_hidden=True,
        trace_layers=trace_layers,
    )
    resolved_trace_layers = generation_trace_config.resolve_trace_layers(len(model.model.transformer.layers))

    tokens_generated = 0

    for idx, generated in enumerate(
        model.generate_tokens_many(
            filtered_prefixes,
            inference_config=config,
            generation_trace_config=generation_trace_config,
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

        assert generated.raw_top_k_token_ids is not None and generated.raw_top_k_token_logits is not None
        assert generated.output_norm_hidden is not None
        layer_hidden_states = None
        if generated.layer_hidden_states is not None:
            layer_hidden_states = rearrange(
                generated.layer_hidden_states[:seqlen],
                "tokens layers hidden -> layers tokens hidden",
            )

        yield TraceCompletionRecord(
            prefix_token_ids=filtered_prefixes[idx],
            completion_token_ids=token_ids,
            raw_topk_token_ids=generated.raw_top_k_token_ids[:seqlen],
            raw_topk_token_logits=generated.raw_top_k_token_logits[:seqlen],
            output_norm_hidden=generated.output_norm_hidden[:seqlen],
            layer_indices=resolved_trace_layers,
            layer_hidden_states=layer_hidden_states,
        )

        if progress_callback is not None:
            progress_callback(CollectTracesEvent(idx + 1, tokens_generated))

        if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
            break
