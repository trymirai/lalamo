import jax.numpy as jnp
import pytest
from jax.lax import DotAlgorithmPreset

from lalamo.inference.batch_scheduler import BatchSchedulerConfig, FixedSizeBatchScheduler
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.models.language_model import GenerationConfig
from lalamo.module import ForwardPassMode, Keychain
from lalamo.modules import DecoderForwardPassConfig
from tests.conftest import ConvertModel, filter_specs, load_converted_model, mark_by_size
from tests.model_test_tiers import ModelTier

core_llm_specs = filter_specs(model_type=LanguageModelSpec, max_tier=ModelTier.CORE)


@pytest.fixture(params=mark_by_size(core_llm_specs), ids=[spec.origin.description for spec in core_llm_specs])
def language_model(request: pytest.FixtureRequest, convert_model: ConvertModel) -> LanguageModel:
    model_dir = convert_model(request.param.origin.description)
    model = load_converted_model(model_dir)
    assert isinstance(model, LanguageModel)
    return model


@pytest.mark.parametrize("num_top_logits_to_return", [None, 8])
def test_eager_generation(language_model: LanguageModel, num_top_logits_to_return: int | None) -> None:
    prompt = [UserMessage("Count from 1 to 10 separated by spaces, using digits.")]
    token_ids = jnp.array(language_model.token_codec.encode_request(prompt))[None, :]
    result = language_model.generate_tokens(
        token_ids,
        max_output_length=1024,
        num_top_logits_to_return=num_top_logits_to_return,
        keychain=Keychain.init(0),
    )
    token_ids = result.token_ids.squeeze(0)
    eos_ids = language_model.config.generation_config.stop_token_ids
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
    token_ids = jnp.array(language_model.token_codec.encode_request(prompt))[None, :]

    response_token_ids = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=jnp.array([0], dtype=jnp.int32),
        max_output_length=1024,
        keychain=Keychain.init(1),
    ).token_ids.squeeze(0)
    response_text = language_model.token_codec.tokenizer.decode(response_token_ids)
    assert "elephants" not in response_text.lower()

    response_token_ids = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=jnp.array([token_ids.size]),
        max_output_length=1024,
        keychain=Keychain.init(2),
    ).token_ids.squeeze(0)
    response_text = language_model.token_codec.tokenizer.decode(response_token_ids)
    assert "elephants" in response_text.lower()


def test_batch_generation(language_model: LanguageModel) -> None:
    prompts = [
        UserMessage("What's the capital of UK?"),
        UserMessage("Talk about apples"),
        UserMessage("Explain why the sky is blue"),
    ]
    inputs = [jnp.array(language_model.token_codec.encode_request([prompt])) for prompt in prompts]
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

    generation_config = GenerationConfig(temperature=0.0)
    max_output_length = 32
    response_token_ids = language_model.generate_tokens(
        padded_token_ids,
        generation_config=generation_config,
        prompt_lengths_without_padding=batched_prompt_lengths,
        max_output_length=max_output_length,
        keychain=Keychain.init(3),
    ).token_ids

    pairs = [(0, 1), (1, 2), (0, 2)]
    outputs: dict[int, list[list[int]]] = {
        prompt_index: [response_token_ids[prompt_index].tolist()] for prompt_index in range(len(prompts))
    }

    for pair_index, (i, j) in enumerate(pairs):
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
            max_output_length=max_output_length,
            keychain=Keychain.init(10 + pair_index),
        ).token_ids

        outputs[i].append(result[0].tolist())
        outputs[j].append(result[1].tolist())

    for prompt_idx, token_lists in outputs.items():
        assert token_lists[0] == token_lists[1], f"Prompt {prompt_idx} produced different outputs in different batches"


def test_streaming_generation(language_model: LanguageModel) -> None:
    prompt = [UserMessage("What's the capital of UK?")]
    token_ids = jnp.array(language_model.token_codec.encode_request(prompt))

    token_stream = language_model.stream_tokens(
        token_ids,
        max_output_length=1024,
        keychain=Keychain.init(4),
    )
    response_token_ids = jnp.array(list(token_stream))
    response_text = language_model.token_codec.tokenizer.decode(response_token_ids)
    assert "london" in response_text.lower(), response_text


def test_streaming_vs_eager_consistency(language_model: LanguageModel) -> None:
    prompt = [UserMessage("What's the largest domestic cat breed?")]
    token_ids = jnp.array(language_model.token_codec.encode_request(prompt))

    generation_config = GenerationConfig(temperature=0.0)
    prefill_forward_pass_config = DecoderForwardPassConfig.for_inference(precision=DotAlgorithmPreset.F32_F32_F32)
    decode_forward_pass_config = DecoderForwardPassConfig.for_inference(
        ForwardPassMode.SINGLE_TOKEN,
        precision=DotAlgorithmPreset.F32_F32_F32,
    )

    generation_keychain = Keychain.init(5)
    max_output_length = 10
    eager_token_ids = language_model.generate_tokens(
        token_ids[None, :],
        generation_config=generation_config,
        max_output_length=max_output_length,
        prefill_forward_pass_config=prefill_forward_pass_config,
        decode_forward_pass_config=decode_forward_pass_config,
        keychain=generation_keychain,
    ).token_ids.squeeze(0)

    streaming_token_generator = language_model.stream_tokens(
        token_ids,
        generation_config=generation_config,
        max_output_length=max_output_length,
        prefill_forward_pass_config=prefill_forward_pass_config,
        decode_forward_pass_config=decode_forward_pass_config,
        keychain=generation_keychain,
    )
    streaming_token_ids = jnp.array(list(streaming_token_generator))

    assert jnp.array_equal(eager_token_ids, streaming_token_ids), (
        eager_token_ids.squeeze().tolist(),
        streaming_token_ids.squeeze().tolist(),
    )

    batch_scheduler = FixedSizeBatchScheduler(model=language_model)
    [(idx, batch_response)] = list(
        batch_scheduler.reply_many(
            [prompt],
            generation_config=generation_config,
            batch_scheduler_config=BatchSchedulerConfig(batch_size=1, max_output_length=10),
            keychain=generation_keychain,
        ),
    )
    assert idx == 0
    streaming_response = language_model.token_codec.decode_response(streaming_token_ids.tolist())
    assert batch_response == streaming_response
