from collections.abc import Iterable

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int

from lalamo.message_processor import Message, UserMessage


def is_non_empty_user_message(message: Message) -> bool:
    if type(message) is not UserMessage:
        return False

    content = message.content
    if isinstance(content, str):
        return bool(content.strip())

    return any(isinstance(block, str) and bool(block.strip()) for block in content)


def get_prefixes_ending_in_user_message(conversation: Iterable[Message]) -> list[list[Message]]:
    conversation = list(conversation)
    return [conversation[: i + 1] for i, msg in enumerate(conversation) if is_non_empty_user_message(msg)]


def pad_sequences(
    sequences: Iterable[list[int]],
    pad_token_id: int,
    padded_length: int,
) -> tuple[Int[Array, "batch tokens"], Bool[Array, "batch tokens"]]:
    sequence_list = list(sequences)
    padded = np.full((len(sequence_list), padded_length), pad_token_id, dtype=np.int32)
    mask = np.zeros((len(sequence_list), padded_length), dtype=np.bool_)

    for index, sequence in enumerate(sequence_list):
        padded[index, : len(sequence)] = np.asarray(sequence, dtype=np.int32)
        mask[index, : len(sequence)] = True

    return jnp.asarray(padded), jnp.asarray(mask)


def make_target_positions(
    prefix_lengths: list[int],
    completion_lengths: list[int],
    max_completion_length: int,
) -> Int[Array, "batch completion_tokens"]:
    positions = np.zeros((len(prefix_lengths), max_completion_length), dtype=np.int32)
    for index, (prefix_length, completion_length) in enumerate(zip(prefix_lengths, completion_lengths, strict=True)):
        positions[index, :completion_length] = prefix_length - 1 + np.arange(completion_length, dtype=np.int32)
    return jnp.asarray(positions)


def round_up_to_multiple(value: int, multiple: int) -> int:
    assert multiple > 0
    return ((value + multiple - 1) // multiple) * multiple
