import jax.numpy as jnp
import pytest

from lalamo.message_processor import UserMessage
from lalamo.model_import.model_specs.common import ModelType
from lalamo.models import LanguageModel
from lalamo.models.common import InferenceConfig
from lalamo.models.language_model import GenerationConfig, LanguageModelConfig
from tests.conftest import ConvertModel, filter_specs, mark_by_size
from tests.model_test_tiers import ModelTier

core_llm_specs = filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.CORE)


@pytest.fixture(params=mark_by_size(core_llm_specs), ids=[spec.repo for spec in core_llm_specs])
def language_model(request: pytest.FixtureRequest, convert_model: ConvertModel) -> LanguageModel:
    model_dir = convert_model(request.param.repo, cached=True)
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
        UserMessage("Explain why the sky is blue"),
    ]
    inputs = [jnp.array(language_model.message_processor.tokenize_request([p])) for p in prompts]
    generation_config = GenerationConfig(temperature=0)

    pairs = [(0, 1), (1, 2), (0, 2)]
    outputs: dict[int, list[list[int]]] = {i: [] for i in range(len(prompts))}

    for i, j in pairs:
        pair_inputs = [inputs[i], inputs[j]]
        max_len = max(inp.size for inp in pair_inputs)
        lengths = jnp.array([inp.size for inp in pair_inputs])
        padded = jnp.array(
            [jnp.pad(inp, (0, max_len - inp.size)) for inp in pair_inputs],
        )

        result = language_model.generate_tokens(
            padded,
            generation_config=generation_config,
            prompt_lengths_without_padding=lengths,
            max_output_length=32,
        ).token_ids

        outputs[i].append(result[0].tolist())
        outputs[j].append(result[1].tolist())

    for prompt_idx, token_lists in outputs.items():
        assert token_lists[0] == token_lists[1], f"Prompt {prompt_idx} produced different outputs in different batches"


def test_streaming_generation(language_model: LanguageModel) -> None:
    prompt = [UserMessage("What's the capital of UK?")]
    token_ids = jnp.array(language_model.message_processor.tokenize_request(prompt))

    token_stream = language_model.stream_tokens(token_ids, max_output_length=32)
    response_token_ids = jnp.array(list(token_stream))
    assert len(response_token_ids) > 0


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
