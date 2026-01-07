import jax.numpy as jnp
import pytest

from lalamo.message_processor import UserMessage
from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.models import LanguageModel
from lalamo.sampling import GreedyPolicy

MODEL_LIST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "LiquidAI/LFM2-700M",
    "cartesia-ai/Llamba-1B",
]


@pytest.fixture(params=MODEL_LIST)
def language_model(request: pytest.FixtureRequest) -> LanguageModel:
    model = import_model(REPO_TO_MODEL[request.param]).model
    assert isinstance(model, LanguageModel)
    return model


@pytest.mark.parametrize("num_top_logits_to_return", [None, 8, 16])
def test_eager_generation(language_model: LanguageModel, num_top_logits_to_return: int | None) -> None:
    prompt = [UserMessage("Count from 1 to 10 separated by spaces.")]
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
    assert response_text == "1 2 3 4 5 6 7 8 9 10", response_text

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
        UserMessage("What's the largest domestic cat breed?"),
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

    response_token_ids = language_model.generate_tokens(
        padded_token_ids,
        prompt_lengths_without_padding=batched_prompt_lengths,
        max_output_length=32,
    ).token_ids

    response_a, response_b = [language_model.message_processor.tokenizer.decode(ids) for ids in response_token_ids]

    assert "london" in response_a.lower() and "maine coon" not in response_a.lower(), response_a
    assert "maine coon" in response_b.lower() and "london" not in response_b.lower(), response_b


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

    sampling_policy = GreedyPolicy()
    eager_token_ids = language_model.generate_tokens(
        token_ids[None, :],
        sampling_policy=sampling_policy,
        max_output_length=32,
        eos_token_ids=jnp.array([-1]),  # Never stop.
    ).token_ids.squeeze(0)

    streaming_token_generator = language_model.stream_tokens(
        token_ids,
        sampling_policy=sampling_policy,
        max_output_length=32,
        eos_token_ids=jnp.array([-1]),  # Never stop.
    )
    streaming_token_ids = jnp.array(list(streaming_token_generator))

    assert jnp.array_equal(eager_token_ids, streaming_token_ids), (
        eager_token_ids.squeeze().tolist(),
        streaming_token_ids.squeeze().tolist(),
    )

    response_text = language_model.message_processor.tokenizer.decode(eager_token_ids)
    assert "maine coon" in response_text.lower(), response_text


pytestmark = pytest.mark.xdist_group("heavy")
