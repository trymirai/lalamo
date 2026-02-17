import re

import jax.numpy as jnp
import pytest

from lalamo.message_processor import UserMessage
from lalamo.model_import import import_model
from lalamo.model_registry import ModelRegistry
from lalamo.models import LanguageModel
from lalamo.models.common import InferenceConfig
from lalamo.models.language_model import GenerationConfig, LanguageModelConfig
from tests.conftest import ConvertModel

MODEL_LIST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "mlx-community/LFM2-2.6B-Exp-8bit",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "cartesia-ai/Llamba-1B",
]


@pytest.fixture(params=MODEL_LIST)
def language_model(request: pytest.FixtureRequest, convert_model: ConvertModel) -> LanguageModel:
    model_dir = convert_model(request.param)
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
    eos_idx = next(i for i, tok in enumerate(token_ids.tolist()) if tok in eos_ids)
    response_text = language_model.message_processor.tokenizer.decode(token_ids[:eos_idx])

    digits_pattern = r"1\s+2\s+3\s+4\s+5\s+6\s+7\s+8\s+9\s+10"
    words_pattern = r"one\s+two\s+three\s+four\s+five\s+six\s+seven\s+eight\s+nine\s+ten"
    assert re.search(digits_pattern, response_text) or re.search(words_pattern, response_text, re.IGNORECASE), (
        response_text
    )

    if num_top_logits_to_return is not None:
        assert result.top_k_token_ids is not None
        assert result.top_k_token_logits is not None

        expected_shape = (1, result.token_ids.shape[1], num_top_logits_to_return)
        assert result.top_k_token_ids.shape == expected_shape
        assert result.top_k_token_logits.shape == expected_shape

        top_k_token_ids = result.top_k_token_ids.squeeze(0).tolist()
        top_k_token_logits = result.top_k_token_logits.squeeze(0).tolist()

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

    generation_config = GenerationConfig(top_k=1)
    response_token_ids = language_model.generate_tokens(
        padded_token_ids,
        generation_config=generation_config,
        prompt_lengths_without_padding=batched_prompt_lengths,
        max_output_length=32,
    ).token_ids

    response_a, response_b = [language_model.message_processor.tokenizer.decode(ids) for ids in response_token_ids]

    assert "london" in response_a.lower() and "apple" not in response_a.lower(), response_a
    assert "apple" in response_b.lower() and "london" not in response_b.lower(), response_b


def test_streaming_vs_eager_consistency(language_model: LanguageModel) -> None:
    prompt = [UserMessage("What's the largest domestic cat breed?")]
    token_ids = jnp.array(language_model.message_processor.tokenize_request(prompt))

    generation_config = GenerationConfig(top_k=1)

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

    [(idx, batch_response)] = list(
        language_model.reply_many(
            [prompt],
            generation_config=generation_config,
            inference_config=InferenceConfig(batch_size=1, max_output_length=10),
        ),
    )
    assert idx == 0
    streaming_response = language_model.message_processor.parse_tokenized_response(streaming_token_ids.tolist())
    assert batch_response == streaming_response
