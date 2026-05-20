from collections.abc import Callable, Iterable
from itertools import batched, chain
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.data.utils import get_prefixes_ending_in_user_message, pad_sequences
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import Message
from lalamo.module import Keychain
from lalamo.speculator.common import Speculator


class CollectTracesEvent(NamedTuple):
    sequences_processed: int
    total_sequences: int | None
    tokens_generated: int


def inference_collect_traces(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    batch_size: int = 1,
    max_input_length: int = 1024,
    max_output_length: int = 4096,
    tokens_to_generate: int | None = None,
    speculator: Speculator | None = None,
    progress_callback: Callable[[CollectTracesEvent], None] | None = None,
) -> Iterable[LalamoCompletion]:
    prefixes = chain.from_iterable(map(get_prefixes_ending_in_user_message, conversations))
    tokenized_prefixes = map(model.token_codec.encode_request, prefixes)
    filtered_prefixes = filter(lambda conv: len(conv) <= max_input_length, tokenized_prefixes)
    tokens_generated = 0
    sequences_processed = 0
    key = jax.random.key(0)

    for prefix_batch in batched(filtered_prefixes, batch_size):
        next_key, batch_key = jax.random.split(key)
        key = next_key
        padded_prefixes, _ = pad_sequences(prefix_batch, 0, max_input_length)
        prefix_lengths = jnp.asarray([len(prefix) for prefix in prefix_batch], dtype=jnp.int32)
        generated_batch = model.generate_tokens(
            padded_prefixes,
            prompt_lengths_without_padding=prefix_lengths,
            max_output_length=max_output_length,
            speculator=speculator,
            keychain=Keychain(vmapped_keys=batch_key, batch_key=batch_key),
        )
        for row_index, prefix_token_ids in enumerate(prefix_batch):
            token_ids = model.trim_at_eos(generated_batch.token_ids[row_index].tolist())
            seqlen = len(token_ids)

            if tokens_to_generate is not None:
                seqlen = min(seqlen, tokens_to_generate - tokens_generated)

            tokens_generated += seqlen
            sequences_processed += 1
            token_ids = token_ids[:seqlen]

            yield LalamoCompletion(
                prefix_token_ids=prefix_token_ids,
                completion_token_ids=token_ids,
            )

            if progress_callback is not None:
                progress_callback(CollectTracesEvent(sequences_processed, None, tokens_generated))

            if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
                return
