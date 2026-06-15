import random

import jax
import pytest

from lalamo.inference.batch_scheduler import (
    BatchSchedulerConfig,
    ContinuousBatchScheduler,
    FixedSizeBatchScheduler,
)
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.models.language_model import GenerationConfig
from lalamo.module import ShardingConfig
from tests.conftest import ConvertModel

_FUZZ_MODEL_REPOS = (
    "Qwen/Qwen3-0.6B",
    "cartesia-ai/Llamba-1B",
)

_FUZZ_PROMPTS = (
    "Say hi.",
    "Name a fruit.",
    "What is 2+2?",
    "Reply with one word.",
    "Yes or no?",
    "Pick a color.",
    "Complete: the sky is",
    "One short word please.",
)

_DP_NUM_DEVICES = 2


@pytest.fixture(scope="module", params=_FUZZ_MODEL_REPOS, ids=_FUZZ_MODEL_REPOS)
def fuzz_language_model(request: pytest.FixtureRequest, _convert_model_session: ConvertModel) -> LanguageModel:
    model_dir = _convert_model_session(request.param, cached=True)
    model = LanguageModel.load(model_dir, sharding_config=ShardingConfig.replicated())
    assert isinstance(model, LanguageModel)
    return model


@pytest.fixture(scope="module", params=_FUZZ_MODEL_REPOS, ids=_FUZZ_MODEL_REPOS)
def fuzz_language_model_dp(request: pytest.FixtureRequest, _convert_model_session: ConvertModel) -> LanguageModel:
    devices = jax.devices()[:_DP_NUM_DEVICES]
    if len(devices) < _DP_NUM_DEVICES:
        pytest.skip(f"requires >= {_DP_NUM_DEVICES} JAX devices for data-parallel test")
    model_dir = _convert_model_session(request.param, cached=True)
    model = LanguageModel.load(model_dir, sharding_config=ShardingConfig.data_parallel(devices=devices))
    assert isinstance(model, LanguageModel)
    return model


def _assert_continuous_matches_fixed(
    model: LanguageModel,
    *,
    seed: int,
    num_prompts: int,
    batch_size: int,
    block_size: int,
    max_output_length: int,
    padded_length: int,
) -> None:
    rng = random.Random(seed)
    prompts = [[UserMessage(rng.choice(_FUZZ_PROMPTS))] for _ in range(num_prompts)]
    tokenized = [model.token_codec.encode_request(prompt) for prompt in prompts]

    generation_config = GenerationConfig(
        temperature=0.0,
        frequency_penalty=0.5,
        stop_token_ids=model.config.generation_config.stop_token_ids,
    )
    batch_scheduler_config = BatchSchedulerConfig(
        batch_size=batch_size,
        max_output_length=max_output_length,
        padded_length=padded_length,
    )

    fixed_results = dict(
        FixedSizeBatchScheduler(model=model).generate_tokens_many(
            tokenized,
            generation_config=generation_config,
            batch_scheduler_config=batch_scheduler_config,
        ),
    )
    continuous_results = dict(
        ContinuousBatchScheduler(model=model, block_size=block_size).generate_tokens_many(
            tokenized,
            generation_config=generation_config,
            batch_scheduler_config=batch_scheduler_config,
        ),
    )

    assert fixed_results.keys() == continuous_results.keys() == set(range(num_prompts))
    for seq_id in range(num_prompts):
        fixed_ids = model.trim_at_eos(fixed_results[seq_id].token_ids.tolist())
        continuous_ids = model.trim_at_eos(continuous_results[seq_id].token_ids.tolist())
        assert fixed_ids == continuous_ids, f"seq {seq_id}: fixed={fixed_ids[:20]} continuous={continuous_ids[:20]}"


@pytest.mark.parametrize(
    ("seed", "num_prompts", "batch_size", "block_size", "max_output_length", "padded_length"),
    [
        # (empty)                            zero sequences
        (0, 0, 2, 8, 8, 32),
        # (under-filled batch, misaligned)   10%8=2, lines stay empty forever
        (1, 1, 4, 8, 10, 48),
        # (exact fit, misaligned)            12%8=4, no refill
        (2, 4, 4, 8, 12, 56),
        # (heavy refill, misaligned)         14%4=2, many churns
        (3, 12, 2, 4, 14, 56),
        # (block_size=1)                     every step is a boundary
        (4, 3, 2, 1, 8, 64),
        # (block > max_output)               block clamps to max_output
        (5, 2, 2, 64, 6, 56),
        # (small batches, misaligned)        batch_size=2, 11%4=3
        (6, 5, 2, 4, 11, 40),
        # (multi-block, misaligned)          batch_size=4, 13%4=1, minimal tail
        (7, 8, 4, 4, 13, 80),
    ],
)
def test_continuous_vs_fixed_fuzz(
    fuzz_language_model: LanguageModel,
    seed: int,
    num_prompts: int,
    batch_size: int,
    block_size: int,
    max_output_length: int,
    padded_length: int,
) -> None:
    _assert_continuous_matches_fixed(
        fuzz_language_model,
        seed=seed,
        num_prompts=num_prompts,
        batch_size=batch_size,
        block_size=block_size,
        max_output_length=max_output_length,
        padded_length=padded_length,
    )


@pytest.mark.parametrize(
    ("seed", "num_prompts", "batch_size", "block_size", "max_output_length", "padded_length"),
    [
        # (under-filled batch)               batch_size divisible by DP mesh
        (1, 1, 4, 8, 10, 48),
        # (heavy refill)                     many churns across the sharded batch axis
        (3, 12, 2, 4, 14, 56),
        # (multi-block)                      several decode blocks with refills
        (7, 8, 4, 4, 13, 80),
    ],
)
def test_continuous_vs_fixed_dp(
    fuzz_language_model_dp: LanguageModel,
    seed: int,
    num_prompts: int,
    batch_size: int,
    block_size: int,
    max_output_length: int,
    padded_length: int,
) -> None:
    _assert_continuous_matches_fixed(
        fuzz_language_model_dp,
        seed=seed,
        num_prompts=num_prompts,
        batch_size=batch_size,
        block_size=block_size,
        max_output_length=max_output_length,
        padded_length=padded_length,
    )
