from collections.abc import Sequence

import jax
import numpy as np

from lalamo.models import GenerationConfig, LanguageModel
from lalamo.module import Keychain, LogicalAxis, ShardingConfig

from .rollouts import Rollouts

__all__ = [
    "ON_POLICY_SAMPLING",
    "pack_rollouts",
    "sample_responses",
]

# Draws from the model's own distribution. Every field holds the value that switches its sampling modifier off,
# because a field left at `None` would keep the model's default instead.
ON_POLICY_SAMPLING = GenerationConfig(
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    min_p=0.0,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
)


def _pad(rows: Sequence[Sequence[int]], length: int) -> np.ndarray:
    longest = max(len(row) for row in rows)
    if longest > length:
        raise ValueError(f"A row of {longest} tokens does not fit into the padded length {length}.")
    return np.asarray([[*row, *([0] * (length - len(row)))] for row in rows], dtype=np.int32)


def _until_stop_token(token_ids: Sequence[int], stop_token_ids: frozenset[int]) -> tuple[int, ...]:
    stop_index = next((index for index, token_id in enumerate(token_ids) if token_id in stop_token_ids), None)
    if stop_index is None:
        return tuple(token_ids)
    return tuple(token_ids[: stop_index + 1])


def sample_responses(
    model: LanguageModel,
    prompts: Sequence[Sequence[int]],
    max_response_length: int,
    generation_config: GenerationConfig,
    *,
    prompt_capacity: int,
    keychain: Keychain,
) -> tuple[tuple[int, ...], ...]:
    """Samples one response per prompt; `generation_config` overrides the model's default sampling settings.

    Each response ends with the stop token that finished it, unless it was cut off at `max_response_length`.
    Prompts are padded to `prompt_capacity`, so calls that share it also share one compiled generation loop.
    """
    sharding_config = model.sharding_config
    batch_axis = sharding_config.resolve_axis(LogicalAxis.BATCH)
    prompt_lengths = np.asarray([len(prompt) for prompt in prompts], dtype=np.int32)
    generated = model.generate_tokens(
        jax.device_put(_pad(prompts, prompt_capacity), sharding_config.make_sharding((batch_axis, None))),
        generation_config=generation_config,
        prompt_lengths_without_padding=jax.device_put(prompt_lengths, sharding_config.make_sharding((batch_axis,))),
        max_output_length=max_response_length,
        keychain=keychain,
    ).token_ids
    stop_token_ids = frozenset(model.config.generation_config.stop_token_ids)
    return tuple(_until_stop_token(row, stop_token_ids) for row in np.asarray(generated).tolist())


def pack_rollouts(
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    *,
    num_tokens: int,
    sharding_config: ShardingConfig,
) -> Rollouts:
    """Joins every prompt with its response; the same responses can be packed behind differently rendered prompts."""
    batch_axis = sharding_config.resolve_axis(LogicalAxis.BATCH)
    token_sharding = sharding_config.make_sharding((batch_axis, None))
    length_sharding = sharding_config.make_sharding((batch_axis,))
    rows = [(*prompt, *response) for prompt, response in zip(prompts, responses, strict=True)]
    prompt_lengths = np.asarray([len(prompt) for prompt in prompts], dtype=np.int32)
    response_lengths = np.asarray([len(response) for response in responses], dtype=np.int32)
    return Rollouts(
        token_ids=jax.device_put(_pad(rows, num_tokens), token_sharding),
        prompt_lengths=jax.device_put(prompt_lengths, length_sharding),
        response_lengths=jax.device_put(response_lengths, length_sharding),
    )
