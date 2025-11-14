import random
from collections.abc import Callable, Iterable
from typing import NamedTuple

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.speculator.common import Speculator


class SpeculatorTrainingEvent(NamedTuple):
    trained_sequences: int
    trained_tokens: int


def train_speculator(
    speculator: Speculator,
    traces: Iterable[LalamoCompletion],
    tokens_to_train: int | None = None,
    progress_callback: Callable[[SpeculatorTrainingEvent], None] | None = None,
) -> None:
    trained_tokens = 0

    for trained_sequences, trace in enumerate(traces, start=1):
        if tokens_to_train is not None and trained_tokens + len(trace.completion_token_ids) > tokens_to_train:
            end = tokens_to_train - trained_tokens
        else:
            end = None
        token_ids = trace.completion_token_ids[:end]
        token_logits = trace.completion_token_logits[:end]

        speculator.train(token_ids, token_logits)

        trained_tokens += len(token_ids)

        if progress_callback is not None:
            progress_callback(SpeculatorTrainingEvent(trained_sequences, trained_tokens))

        if tokens_to_train is not None and trained_tokens >= tokens_to_train:
            break


def test_speculator(
    speculator: Speculator,
    sequence: Iterable[int] = [],
    max_completion_length: int = 32,
) -> list[int]:
    sequence = list(sequence)
    for _ in range(max_completion_length):
        probs = speculator.probs(sequence)
        if sum(probs.values()) == 0:
            break

        selected = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        sequence.append(selected)
    return sequence
