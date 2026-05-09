from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import jax.numpy as jnp
import numpy as np
from annotated_types import Ge
from jaxtyping import Array, Bool, Float, Int

from lalamo.data.lalamo_completions import LalamoCompletion

type LayerSelection = Literal["all"] | tuple[int, ...]
type PositionSelection = Literal["all"]


@dataclass(frozen=True)
class FeatureRequest:
    completions: Iterable[LalamoCompletion]
    batch_size: Annotated[int, Ge(1)] = 1
    layers: LayerSelection = tuple()
    positions: PositionSelection = "all"
    k: int = 8
    top_k_logits: Annotated[int, Ge(1)] = 128
    output_features: bool = False
    pad_token_id: int = 0
    prompt_padding_multiple: int = 128
    generation_padding_multiple: int = 512


@dataclass(frozen=True)
class LalamoCompletionBatch:
    prefix_token_ids: Int[Array, "batch prefix_tokens"]
    prefix_mask: Bool[Array, "batch prefix_tokens"]
    completion_token_ids: Int[Array, "batch completion_tokens"]
    completion_mask: Bool[Array, "batch completion_tokens"]
    input_token_ids: Int[Array, "batch input_tokens"]
    input_lengths: Int[Array, " batch"]
    target_token_ids: Int[Array, "batch completion_tokens"]
    target_mask: Bool[Array, "batch completion_tokens"]
    target_positions: Int[Array, "batch completion_tokens"]

    @classmethod
    def from_completions(
        cls,
        completions: Iterable[LalamoCompletion],
        pad_token_id: int = 0,
        prompt_padding_multiple: int = 128,
        generation_padding_multiple: int = 512,
    ) -> Self:
        completion_list = list(completions)
        if not completion_list:
            raise ValueError("completion batch must not be empty.")

        prefix_sequences = [completion.prefix_token_ids for completion in completion_list]
        completion_sequences = [completion.completion_token_ids for completion in completion_list]
        input_sequences = [
            [*completion.prefix_token_ids, *completion.completion_token_ids[:-1]] for completion in completion_list
        ]
        prompt_length = round_up_to_multiple(
            max(len(sequence) for sequence in prefix_sequences),
            prompt_padding_multiple,
        )
        generation_length = round_up_to_multiple(
            max(len(sequence) for sequence in completion_sequences),
            generation_padding_multiple,
        )
        prefix_token_ids, prefix_mask = pad_sequences(prefix_sequences, pad_token_id, prompt_length)
        completion_token_ids, completion_mask = pad_sequences(completion_sequences, pad_token_id, generation_length)
        input_token_ids, _ = pad_sequences(
            input_sequences,
            pad_token_id,
            prompt_length + max(generation_length - 1, 0),
        )
        input_lengths = jnp.asarray([len(sequence) for sequence in input_sequences], dtype=jnp.int32)
        target_positions = make_target_positions(
            [len(completion.prefix_token_ids) for completion in completion_list],
            [len(completion.completion_token_ids) for completion in completion_list],
            generation_length,
        )

        return cls(
            prefix_token_ids=prefix_token_ids,
            prefix_mask=prefix_mask,
            completion_token_ids=completion_token_ids,
            completion_mask=completion_mask,
            input_token_ids=input_token_ids,
            input_lengths=input_lengths,
            target_token_ids=completion_token_ids,
            target_mask=completion_mask,
            target_positions=target_positions,
        )


@dataclass(frozen=True)
class LalamoCompletionFeatures:
    completion_batch: LalamoCompletionBatch
    target_logsumexp: Float[Array, "batch completion_tokens"] | None = None
    target_top_k_ids: Int[Array, "batch completion_tokens k"] | None = None
    target_top_k_logits: Float[Array, "batch completion_tokens k"] | None = None
    output_features: Float[Array, "batch completion_tokens hidden"] | None = None
    layer_features: Float[Array, "batch layers completion_tokens hidden"] | None = None


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
