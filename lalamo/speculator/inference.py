import functools
from collections.abc import Callable, Iterable
from itertools import batched, chain, islice
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax._src.stages import Compiled

from lalamo.common import decrease_batchsize_on_oom
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
    def make_generate_tokens_compiled(batch_size: int) -> Compiled:
        return (
            jax.jit(
                functools.partial(
                    LanguageModel.generate_tokens,
                    max_output_length=max_output_length,
                    num_top_logits_to_return=num_top_logits_to_collect,
                ),
            )
            .lower(
                model,
                prompt_token_ids=jax.ShapeDtypeStruct((batch_size, max_input_length), jnp.int32),
                prompt_lengths_without_padding=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            )
            # the autotune levels are (according to https://guides.lw1.at/all-xla-options/#--xla_gpu_autotune_level)
            # 0 - no autotune, gpu shouldn't be touched
            # 1 - basic level, gpu should be touched veeery little
            # 2,3 - gpu touched more and more
            # 4 (default) - gpu might allocate more memory than the run would require!
            .compile(compiler_options={"xla_gpu_autotune_level": "0"})
        )

    prefixes = chain.from_iterable(map(get_prefixes_ending_in_user_message, conversations))

    tokenized_prefixes = map(model.message_processor.tokenize_request, prefixes)
    filtered_prefixes = filter(lambda conv: len(conv) <= max_input_length, tokenized_prefixes)

    test_batch = list(islice(filtered_prefixes, batch_size))

    def collect_traces_body(batch_size: int) -> Iterable[LalamoCompletion]:
        tokens_generated, sequences_processed = 0, 0
        generate_tokens_compiled = make_generate_tokens_compiled(batch_size)
        for real_batch in batched(chain(test_batch, filtered_prefixes), n=batch_size):
            batch_padding = batch_size - len(real_batch)
            batch = (*real_batch, *(([0],) * batch_padding))

            length_without_padding = jnp.array(list(map(len, batch)))

            padded = jnp.array(
                [
                    jnp.pad(jnp.array(tokens), (0, max_input_length - len(tokens)), constant_values=0)
                    for tokens in batch
                ],
            )

            generated = generate_tokens_compiled(
                model,
                prompt_token_ids=padded,
                prompt_lengths_without_padding=length_without_padding,
            )

            assert generated.top_k_token_ids is not None and generated.top_k_token_logits is not None

            for conv_idx in range(len(real_batch)):
                token_ids = generated.token_ids[conv_idx].tolist()
                seqlen = next((i + 1 for i, t in enumerate(token_ids) if t in model.stop_token_ids), len(token_ids))
                if tokens_to_generate is not None:
                    seqlen = min(seqlen, tokens_to_generate - tokens_generated)
                tokens_generated += seqlen
                sequences_processed += 1

                token_ids = token_ids[:seqlen]
                token_logits_ids = generated.top_k_token_ids[conv_idx, : len(token_ids)].tolist()
                token_logits_values = generated.top_k_token_logits[conv_idx, : len(token_ids)].tolist()
                token_logits = [
                    dict(zip(keys, values, strict=True))
                    for keys, values in zip(token_logits_ids, token_logits_values, strict=True)
                ]

                yield LalamoCompletion(batch[conv_idx], token_ids, token_logits)

                if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
                    break

            if progress_callback is not None:
                progress_callback(CollectTracesEvent(sequences_processed, tokens_generated))

            if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
                break

    yield from decrease_batchsize_on_oom(collect_traces_body, batch_size)
