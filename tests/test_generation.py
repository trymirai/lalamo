from textwrap import dedent

import jax.numpy as jnp
import pytest
from attr import dataclass
from jax import jit, vmap
from jaxtyping import Array, Int
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from lalamo.language_model import LanguageModel
from lalamo.model_import import REPO_TO_MODEL, import_model


@dataclass
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
    decoder = import_model(REPO_TO_MODEL["Qwen/Qwen3-0.6B"]).model
    return LanguageModel(decoder)


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_eager_generation(
    language_model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    generation_input: GenerationInput,
) -> None:
    response_token_ids = language_model.generate(generation_input.token_ids, max_output_length=32)
    response_text = tokenizer.decode(response_token_ids)
    assert "<|im_end|>" in response_text


def test_padding(language_model: LanguageModel, tokenizer: PreTrainedTokenizer) -> None:
    prompt = dedent(
        """
        <|im_start|>user
        Talk about elephants<|im_end|>
        <|im_start|>assistant
        <think>\n\n</think>\n\n
    """.lstrip(),
    )
    token_ids = jnp.array(tokenizer.encode(prompt))

    length = jnp.array(0, dtype=jnp.int32)
    response_token_ids = language_model.generate(
        token_ids,
        prompt_length_without_padding=length,
        max_output_length=32,
    )
    response_text = tokenizer.decode(response_token_ids)
    assert "elephants" not in response_text.lower()

    response_token_ids = language_model.generate(
        token_ids,
        prompt_length_without_padding=token_ids.size,
        max_output_length=32,
    )
    response_text = tokenizer.decode(response_token_ids)
    assert "elephants" in response_text.lower()


def test_jit_generation(
    language_model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    generation_input: GenerationInput,
) -> None:
    response_token_ids = jit(language_model.generate, static_argnames=["max_output_length"])(
        generation_input.token_ids,
        max_output_length=32,
    )
    response_text = tokenizer.decode(response_token_ids)
    assert "<|im_end|>" in response_text


def test_batch_generation(
    language_model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    generation_input: GenerationInput,
    another_generation_input: GenerationInput,
) -> None:
    def generate_fn(token_ids: Int[Array, " tokens"], sequence_length: Int[Array, ""]) -> Int[Array, " tokens"]:
        return language_model.generate(
            token_ids,
            prompt_length_without_padding=sequence_length,
            max_output_length=32,
        )

    batched_generate_fn = jit(vmap(generate_fn))

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

    response_token_ids = batched_generate_fn(padded_token_ids, batched_prompt_lengths)
    for ids in response_token_ids:
        response_text = tokenizer.decode(ids)
        assert "<|im_end|>" in response_text


def test_streaming_generation(
    language_model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    generation_input: GenerationInput,
) -> None:
    token_stream = language_model.stream(generation_input.token_ids, max_output_length=32)
    response_token_ids = jnp.array(list(token_stream))
    response_text = tokenizer.decode(response_token_ids)
    assert "<|im_end|>" in response_text


def test_streaming_vs_eager_consistency(
    language_model: LanguageModel,
    generation_input: GenerationInput,
) -> None:
    eager_token_ids = language_model.generate(
        generation_input.token_ids,
        max_output_length=32,
        eos_token_ids=jnp.array([-1]),  # Never stop.
    )

    streaming_token_generator = language_model.stream(
        generation_input.token_ids,
        max_output_length=32,
        eos_token_ids=jnp.array([-1]),  # Never stop.
    )
    streaming_token_ids = jnp.array(list(streaming_token_generator))

    assert jnp.array_equal(eager_token_ids, streaming_token_ids)
