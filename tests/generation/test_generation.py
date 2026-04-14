import random

import jax.numpy as jnp
import pytest

from lalamo.message_processor import UserMessage
from lalamo.model_import.model_specs.common import ModelType
from lalamo.models import ContinuousBatchScheduler, FixedBatchScheduler, LanguageModel
from lalamo.models.common import InferenceConfig
from lalamo.models.language_model import GenerationConfig, LanguageModelConfig
from tests.conftest import ConvertModel, filter_specs, mark_by_size
from tests.model_test_tiers import ModelTier

core_llm_specs = filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.CORE)


@pytest.fixture(params=mark_by_size(core_llm_specs), ids=[spec.repo for spec in core_llm_specs])
def language_model(request: pytest.FixtureRequest, convert_model: ConvertModel) -> LanguageModel:
    model_dir = convert_model(request.param.repo)
    return LanguageModelConfig.load_model(model_dir)


@pytest.mark.parametrize("num_top_logits_to_return", [None, 8, 16])
def test_eager_generation(language_model: LanguageModel, num_top_logits_to_return: int | None) -> None:
    prompt = [UserMessage("Count from 1 to 10 separated by spaces, using digits.")]
    token_ids = jnp.array(language_model.message_processor.tokenize_request(prompt))[None, :]
    result = language_model.generate_tokens(
        token_ids,
        max_output_length=32,
        num_top_logits_to_return=num_top_logits_to_return,
    )
    token_ids = result.token_ids.squeeze(0)
    eos_ids = language_model.stop_token_ids
    eos_idx = next((i for i, tok in enumerate(token_ids.tolist()) if tok in eos_ids), None)
    if num_top_logits_to_return is not None:
        assert result.top_k_token_ids is not None
        assert result.top_k_token_logits is not None

        expected_shape = (1, result.token_ids.shape[1], num_top_logits_to_return)
        assert result.top_k_token_ids.shape == expected_shape
        assert result.top_k_token_logits.shape == expected_shape

        top_k_token_ids = result.top_k_token_ids.squeeze(0).tolist()
        top_k_token_logits = result.top_k_token_logits.squeeze(0).tolist()

        if eos_idx is not None:
            assert top_k_token_ids[eos_idx][0] in eos_ids
            assert top_k_token_logits[eos_idx][0] > max(top_k_token_logits[eos_idx][1:])
    else:
        assert result.top_k_token_ids is None
        assert result.top_k_token_logits is None


def test_padding(language_model: LanguageModel) -> None:
    prompt = [UserMessage("Talk about elephants")]
    token_ids = jnp.array(language_model.message_processor.tokenize_request(prompt))[None, :]

    response_token_ids = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=jnp.array([0], dtype=jnp.int32),
        max_output_length=32,
    ).token_ids.squeeze(0)
    response_text = language_model.message_processor.tokenizer.decode(response_token_ids)
    assert "elephants" not in response_text.lower()

    response_token_ids = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=jnp.array([token_ids.size]),
        max_output_length=32,
    ).token_ids.squeeze(0)
    response_text = language_model.message_processor.tokenizer.decode(response_token_ids)
    assert "elephants" in response_text.lower()


def test_batch_generation(language_model: LanguageModel) -> None:
    prompts = [
        UserMessage("What's the capital of UK?"),
        UserMessage("Talk about apples"),
    ]
    inputs = [jnp.array(language_model.message_processor.tokenize_request([prompt])) for prompt in prompts]
    pad_token_id = 0

    max_len = max(inp.size for inp in inputs)
    batched_prompt_lengths = jnp.array([inp.size for inp in inputs])
    padded_token_ids = jnp.array(
        [
            jnp.pad(
                inp,
                (0, max_len - inp.size),
                constant_values=pad_token_id,
            )
            for inp in inputs
        ],
    )

    generation_config = GenerationConfig(temperature=0)
    response_token_ids = language_model.generate_tokens(
        padded_token_ids,
        generation_config=generation_config,
        prompt_lengths_without_padding=batched_prompt_lengths,
        max_output_length=32,
    ).token_ids

    response_a, response_b = [language_model.message_processor.tokenizer.decode(ids) for ids in response_token_ids]

    assert "london" in response_a.lower() and "apple" not in response_a.lower(), response_a
    assert "apple" in response_b.lower() and "london" not in response_b.lower(), response_b


def test_streaming_generation(language_model: LanguageModel) -> None:
    prompt = [UserMessage("What's the capital of UK?")]
    token_ids = jnp.array(language_model.message_processor.tokenize_request(prompt))

    token_stream = language_model.stream_tokens(token_ids, max_output_length=32)
    response_token_ids = jnp.array(list(token_stream))
    response_text = language_model.message_processor.tokenizer.decode(response_token_ids)
    assert "london" in response_text.lower(), response_text


def test_streaming_vs_eager_consistency(language_model: LanguageModel) -> None:
    prompt = [UserMessage("What's the largest domestic cat breed?")]
    token_ids = jnp.array(language_model.message_processor.tokenize_request(prompt))

    generation_config = GenerationConfig(temperature=0)

    eager_token_ids = language_model.generate_tokens(
        token_ids[None, :],
        generation_config=generation_config,
        max_output_length=10,
    ).token_ids.squeeze(0)

    streaming_token_generator = language_model.stream_tokens(
        token_ids,
        generation_config=generation_config,
        max_output_length=10,
    )
    streaming_token_ids = jnp.array(list(streaming_token_generator))

    assert jnp.array_equal(eager_token_ids, streaming_token_ids), (
        eager_token_ids.squeeze().tolist(),
        streaming_token_ids.squeeze().tolist(),
    )

    scheduler = FixedBatchScheduler(model=language_model)
    [(idx, batch_response)] = list(
        scheduler.reply_many(
            [prompt],
            generation_config=generation_config,
            inference_config=InferenceConfig(batch_size=1, max_output_length=10),
        ),
    )
    assert idx == 0
    streaming_response = language_model.message_processor.parse_tokenized_response(streaming_token_ids.tolist())
    assert batch_response == streaming_response


_FUZZ_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

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


@pytest.fixture(scope="module")
def fuzz_language_model(_convert_model_session: ConvertModel) -> LanguageModel:
    model_dir = _convert_model_session(_FUZZ_MODEL_REPO, cached=True)
    return LanguageModelConfig.load_model(model_dir)


@pytest.mark.parametrize(
    ("seed", "num_prompts", "batch_size", "block_size", "max_output_length", "padded_length"),
    [
        # (empty)                            zero sequences
        (0, 0, 2, 8, 8, 32),
        # (under-filled batch, misaligned)   10%8=2, lines stay empty forever
        (1, 1, 4, 8, 10, 48),
        # (exact fit, misaligned)            12%8=4, no refill
        (2, 4, 4, 8, 12, 32),
        # (heavy refill, misaligned)         14%4=2, many churns
        (3, 12, 2, 4, 14, 40),
        # (block_size=1)                     every step is a boundary
        (4, 3, 2, 1, 8, 64),
        # (block > max_output)               block clamps to max_output
        (5, 2, 2, 64, 6, 56),
        # (sequential, misaligned)           batch_size=1, 11%4=3
        (6, 5, 1, 4, 11, 40),
        # (multi-block, misaligned)          13%4=1, minimal tail
        (7, 8, 3, 4, 13, 80),
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
    rng = random.Random(seed)
    prompts = [[UserMessage(rng.choice(_FUZZ_PROMPTS))] for _ in range(num_prompts)]
    tokenized = [fuzz_language_model.message_processor.tokenize_request(p) for p in prompts]

    generation_config = GenerationConfig(
        temperature=0,
        stop_token_ids=fuzz_language_model.config.generation_config.stop_token_ids,
    )
    inference_config = InferenceConfig(
        batch_size=batch_size,
        max_output_length=max_output_length,
        padded_length=padded_length,
    )

    fixed_results = dict(
        FixedBatchScheduler(model=fuzz_language_model).generate_tokens_many(
            tokenized,
            generation_config=generation_config,
            inference_config=inference_config,
        ),
    )
    continuous_results = dict(
        ContinuousBatchScheduler(
            model=fuzz_language_model,
            continuous_batching_block_size=block_size,
        ).generate_tokens_many(
            tokenized,
            generation_config=generation_config,
            inference_config=inference_config,
        ),
    )

    assert fixed_results.keys() == continuous_results.keys() == set(range(num_prompts))
    for seq_id in range(num_prompts):
        fixed_ids = fuzz_language_model.trim_at_eos(fixed_results[seq_id].token_ids.tolist())
        continuous_ids = fuzz_language_model.trim_at_eos(continuous_results[seq_id].token_ids.tolist())
        assert fixed_ids == continuous_ids, f"seq {seq_id}: fixed={fixed_ids[:20]} continuous={continuous_ids[:20]}"
