from textwrap import dedent

import jax.numpy as jnp
import pytest
from attr import dataclass
from jaxtyping import Array, Int
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.models import LanguageModel
from lalamo.sampling import GreedyPolicy


@dataclass(frozen=True)
class GenerationInput:
    prompt: str
    token_ids: Int[Array, " tokens"]


@pytest.fixture
def generation_input(tokenizer: PreTrainedTokenizer) -> GenerationInput:
    prompt = dedent(
        """
        <|im_start|>user
        hello<|im_end|>
        <|im_start|>assistant
        <think>\n\n</think>\n\n
    """.lstrip(),
    )
    tokens = jnp.array(tokenizer.encode(prompt))
    return GenerationInput(prompt, tokens)


@pytest.fixture
def another_generation_input(tokenizer: PreTrainedTokenizer) -> GenerationInput:
    prompt = dedent(
        """
        <|im_start|>user
        How are you?<|im_end|>
        <|im_start|>assistant
        <think>\n\n</think>\n\n
    """.lstrip(),
    )
    tokens = jnp.array(tokenizer.encode(prompt))
    return GenerationInput(prompt, tokens)


@pytest.fixture
def language_model() -> LanguageModel:
    model = import_model(REPO_TO_MODEL["Qwen/Qwen3-0.6B"]).model
    assert isinstance(model, LanguageModel)
    return model


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_tokenizer(language_model: LanguageModel, generation_input: GenerationInput) -> None:
    token_ids = language_model.message_processor.tokenize_text(generation_input.prompt)
    ref_token_ids = generation_input.token_ids.tolist()
    assert token_ids == ref_token_ids


@pytest.mark.parametrize("num_top_logits_to_return", [None, 8, 16])
def test_eager_generation(
    language_model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    generation_input: GenerationInput,
    num_top_logits_to_return: int | None,
) -> None:
    result = language_model.generate_tokens(
        generation_input.token_ids[None, :],
        max_output_length=32,
        num_top_logits_to_return=num_top_logits_to_return,
    )
    token_ids = result.token_ids.squeeze(0)
    response_text = tokenizer.decode(token_ids)  # type: ignore
    assert "<|im_end|>" in response_text

    if num_top_logits_to_return is not None:
        assert result.top_k_token_ids is not None
        assert result.top_k_token_logits is not None

        expected_shape = (1, result.token_ids.shape[1], num_top_logits_to_return)
        assert result.top_k_token_ids.shape == expected_shape
        assert result.top_k_token_logits.shape == expected_shape

        top_k_token_ids = result.top_k_token_ids.squeeze(0).tolist()
        top_k_token_logits = result.top_k_token_logits.squeeze(0).tolist()

        eos_id = tokenizer.encode("<|im_end|>")[0]
        eos_idx = token_ids.tolist().index(eos_id)
        assert top_k_token_ids[eos_idx][0] == eos_id
        assert top_k_token_logits[eos_idx][0] > max(top_k_token_logits[eos_idx][1:])
    else:
        assert result.top_k_token_ids is None
        assert result.top_k_token_logits is None


def test_padding(language_model: LanguageModel, tokenizer: PreTrainedTokenizer) -> None:
    prompt = dedent(
        """
        <|im_start|>user
        Talk about elephants<|im_end|>
        <|im_start|>assistant
        <think>\n\n</think>\n\n
    """.lstrip(),
    )
    token_ids = jnp.array(tokenizer.encode(prompt))[None, :]

    response_token_ids = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=jnp.array([0], dtype=jnp.int32),
        max_output_length=32,
    ).token_ids.squeeze(0)
    response_text = tokenizer.decode(response_token_ids)  # type: ignore
    assert "elephants" not in response_text.lower()

    response_token_ids = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=jnp.array([token_ids.size]),
        max_output_length=32,
    ).token_ids.squeeze(0)
    response_text = tokenizer.decode(response_token_ids)  # type: ignore
    assert "elephants" in response_text.lower()


def test_batch_generation(
    language_model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    generation_input: GenerationInput,
    another_generation_input: GenerationInput,
) -> None:
    inputs = [generation_input, another_generation_input]
    pad_token_id = 0

    max_len = max(inp.token_ids.size for inp in inputs)

    batched_prompt_lengths = jnp.array([inp.token_ids.size for inp in inputs])
    padded_token_ids = jnp.array(
        [
            jnp.pad(
                inp.token_ids,
                (0, max_len - inp.token_ids.size),
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
    for ids in response_token_ids:
        response_text = tokenizer.decode(ids)
        assert "<|im_end|>" in response_text


def test_streaming_generation(
    language_model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    generation_input: GenerationInput,
) -> None:
    token_stream = language_model.stream_tokens(generation_input.token_ids, max_output_length=32)
    response_token_ids = jnp.array(list(token_stream))
    response_text = tokenizer.decode(response_token_ids)  # type: ignore
    assert "<|im_end|>" in response_text


def test_streaming_vs_eager_consistency(
    language_model: LanguageModel,
    generation_input: GenerationInput,
) -> None:
    sampling_policy = GreedyPolicy()
    eager_token_ids = language_model.generate_tokens(
        generation_input.token_ids[None, :],
        sampling_policy=sampling_policy,
        max_output_length=32,
        eos_token_ids=jnp.array([-1]),  # Never stop.
    ).token_ids.squeeze(0)

    streaming_token_generator = language_model.stream_tokens(
        generation_input.token_ids,
        sampling_policy=sampling_policy,
        max_output_length=32,
        eos_token_ids=jnp.array([-1]),  # Never stop.
    )
    streaming_token_ids = jnp.array(list(streaming_token_generator))

    assert jnp.array_equal(eager_token_ids, streaming_token_ids), (
        eager_token_ids.squeeze().tolist(),
        streaming_token_ids.squeeze().tolist(),
    )
