from collections.abc import Generator
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.lax import DotAlgorithmPreset

from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.models.language_model import _COMPILED_PROMPT_LENGTHS, GenerationConfig
from lalamo.module import ForwardPassMode, Keychain, LogicalAxis, ShardingConfig
from lalamo.modules import DecoderForwardPassConfig
from tests.conftest import ConvertModel, filter_specs, load_converted_model, mark_by_size
from tests.model_test_tiers import ModelTier

core_llm_specs = filter_specs(model_type=LanguageModelSpec, max_tier=ModelTier.CORE)


def _stable_generation_forward_pass_configs() -> tuple[DecoderForwardPassConfig, DecoderForwardPassConfig]:
    prefill_forward_pass_config = DecoderForwardPassConfig.for_tracer_tests()
    decode_forward_pass_config = DecoderForwardPassConfig.for_tracer_tests()
    decode_transformer_config = decode_forward_pass_config.transformer_forward_pass_config
    decode_forward_pass_config = replace(
        decode_forward_pass_config,
        transformer_forward_pass_config=replace(
            decode_transformer_config,
            mlp_forward_pass_config=replace(
                decode_transformer_config.mlp_forward_pass_config,
                mode=ForwardPassMode.SINGLE_TOKEN,
            ),
        ),
    )
    return prefill_forward_pass_config, decode_forward_pass_config


def _batch_axis_size(language_model: LanguageModel) -> int:
    batch_axis = language_model.sharding_config.resolve_axis(LogicalAxis.BATCH)
    if batch_axis is None:
        return 1
    return language_model.sharding_config.mesh.shape[batch_axis]


def _sharded_generation_batch(
    language_model: LanguageModel,
    token_ids: jax.Array,
    lengths_without_padding: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch_size, sequence_length = token_ids.shape
    batch_axis_size = _batch_axis_size(language_model)
    remainder = batch_size % batch_axis_size
    if remainder == 0:
        padded_token_ids = token_ids
        padded_lengths = lengths_without_padding
    else:
        num_padding_rows = batch_axis_size - remainder
        padded_token_ids = jnp.concatenate(
            [
                token_ids,
                jnp.zeros((num_padding_rows, sequence_length), dtype=token_ids.dtype),
            ],
        )
        padded_lengths = jnp.concatenate(
            [
                lengths_without_padding,
                jnp.ones((num_padding_rows,), dtype=lengths_without_padding.dtype),
            ],
        )

    batch_axis = language_model.sharding_config.resolve_axis(LogicalAxis.BATCH)
    return (
        jax.device_put(
            padded_token_ids,
            language_model.sharding_config.make_sharding((batch_axis, None)),
        ),
        jax.device_put(
            padded_lengths,
            language_model.sharding_config.make_sharding((batch_axis,)),
        ),
    )


def _take_batch_prefix(language_model: LanguageModel, values: jax.Array, batch_size: int) -> np.ndarray:
    full_mesh_replicated_sharding = language_model.sharding_config.make_sharding((None,) * values.ndim)
    prefix = values.at[:batch_size].get(out_sharding=full_mesh_replicated_sharding)
    return np.asarray(prefix)


def _take_first_batch_row(language_model: LanguageModel, values: jax.Array) -> np.ndarray:
    return _take_batch_prefix(language_model, values, 1).squeeze(0)


@pytest.fixture(params=mark_by_size(core_llm_specs), ids=[spec.origin.description for spec in core_llm_specs])
def language_model(request: pytest.FixtureRequest, convert_model: ConvertModel) -> Generator[LanguageModel]:
    model_dir = convert_model(request.param.origin.description, cached=True)
    model = load_converted_model(model_dir, ShardingConfig.replicated())
    assert isinstance(model, LanguageModel)
    with jax.set_mesh(model.sharding_config.mesh):
        yield model


@pytest.fixture(params=mark_by_size(core_llm_specs), ids=[spec.origin.description for spec in core_llm_specs])
def replicated_language_model(request: pytest.FixtureRequest, convert_model: ConvertModel) -> LanguageModel:
    model_dir = convert_model(request.param.origin.description, cached=True)
    model = load_converted_model(model_dir, ShardingConfig.replicated())
    assert isinstance(model, LanguageModel)
    return model


@pytest.mark.parametrize("num_top_logits_to_return", [None, 8])
def test_eager_generation(language_model: LanguageModel, num_top_logits_to_return: int | None) -> None:
    prompt = [UserMessage("Count from 1 to 10 separated by spaces, using digits.")]
    raw_token_ids = jnp.array(language_model.token_codec.encode_request(prompt))[None, :]
    _, prompt_length = raw_token_ids.shape
    token_ids, prompt_lengths = _sharded_generation_batch(
        language_model,
        raw_token_ids,
        jnp.array([prompt_length], dtype=jnp.int32),
    )
    result = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=prompt_lengths,
        max_output_length=64,
        num_top_logits_to_return=num_top_logits_to_return,
        keychain=Keychain.init(0, sharding_config=language_model.sharding_config),
    )
    token_ids = _take_first_batch_row(language_model, result.token_ids)
    eos_ids = language_model.config.generation_config.stop_token_ids
    eos_idx = next((i for i, tok in enumerate(token_ids.tolist()) if tok in eos_ids), None)
    if num_top_logits_to_return is not None:
        assert result.top_k_token_ids is not None
        assert result.top_k_token_logits is not None

        generation_batch_size, response_length = result.token_ids.shape
        expected_shape = (generation_batch_size, response_length, num_top_logits_to_return)
        assert result.top_k_token_ids.shape == expected_shape
        assert result.top_k_token_logits.shape == expected_shape

        top_k_token_ids = _take_first_batch_row(language_model, result.top_k_token_ids).tolist()
        top_k_token_logits = _take_first_batch_row(language_model, result.top_k_token_logits).tolist()

        if eos_idx is not None:
            assert top_k_token_ids[eos_idx][0] in eos_ids
            assert top_k_token_logits[eos_idx][0] > max(top_k_token_logits[eos_idx][1:])
    else:
        assert result.top_k_token_ids is None
        assert result.top_k_token_logits is None


def test_padding(language_model: LanguageModel) -> None:
    prompt = [UserMessage("Talk about elephants")]
    raw_token_ids = jnp.array(language_model.token_codec.encode_request(prompt))[None, :]

    token_ids, prompt_lengths = _sharded_generation_batch(
        language_model,
        raw_token_ids,
        jnp.array([0], dtype=jnp.int32),
    )
    response_token_ids = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=prompt_lengths,
        max_output_length=64,
        keychain=Keychain.init(1, sharding_config=language_model.sharding_config),
    ).token_ids
    response_token_ids = _take_first_batch_row(language_model, response_token_ids)
    response_text = language_model.token_codec.tokenizer.decode(response_token_ids.tolist())
    assert "elephants" not in response_text.lower()

    token_ids, prompt_lengths = _sharded_generation_batch(
        language_model,
        raw_token_ids,
        jnp.array([raw_token_ids.size], dtype=jnp.int32),
    )
    response_token_ids = language_model.generate_tokens(
        token_ids,
        prompt_lengths_without_padding=prompt_lengths,
        max_output_length=64,
        keychain=Keychain.init(2, sharding_config=language_model.sharding_config),
    ).token_ids
    response_token_ids = _take_first_batch_row(language_model, response_token_ids)
    response_text = language_model.token_codec.tokenizer.decode(response_token_ids.tolist())
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
    raw_padded_token_ids = jnp.array(
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
    prefill_forward_pass_config, decode_forward_pass_config = _stable_generation_forward_pass_configs()
    max_output_length = 10
    padded_token_ids, batched_prompt_lengths = _sharded_generation_batch(
        language_model,
        raw_padded_token_ids,
        batched_prompt_lengths,
    )
    response_token_ids = language_model.generate_tokens(
        padded_token_ids,
        generation_config=generation_config,
        prompt_lengths_without_padding=batched_prompt_lengths,
        max_output_length=max_output_length,
        prefill_forward_pass_config=prefill_forward_pass_config,
        decode_forward_pass_config=decode_forward_pass_config,
        keychain=Keychain.init(3, sharding_config=language_model.sharding_config),
    ).token_ids
    response_token_ids = _take_batch_prefix(language_model, response_token_ids, len(prompts))

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

        padded, lengths = _sharded_generation_batch(language_model, padded, lengths)

        result = language_model.generate_tokens(
            padded,
            generation_config=generation_config,
            prompt_lengths_without_padding=lengths,
            max_output_length=max_output_length,
            prefill_forward_pass_config=prefill_forward_pass_config,
            decode_forward_pass_config=decode_forward_pass_config,
            keychain=Keychain.init(10 + pair_index, sharding_config=language_model.sharding_config),
        ).token_ids
        result = _take_batch_prefix(language_model, result, 2)

        outputs[i].append(result[0].tolist())
        outputs[j].append(result[1].tolist())

    for prompt_idx, token_lists in outputs.items():
        assert token_lists[0] == token_lists[1], f"Prompt {prompt_idx} produced different outputs in different batches"


def test_streaming_vs_eager_consistency(replicated_language_model: LanguageModel) -> None:
    prompt = [UserMessage("What's the largest domestic cat breed?")]
    token_ids = jnp.array(replicated_language_model.token_codec.encode_request(prompt))

    generation_config = GenerationConfig(temperature=0.0)
    prefill_forward_pass_config = DecoderForwardPassConfig.for_inference(precision=DotAlgorithmPreset.F32_F32_F32)
    decode_forward_pass_config = DecoderForwardPassConfig.for_inference(
        ForwardPassMode.SINGLE_TOKEN,
        precision=DotAlgorithmPreset.F32_F32_F32,
    )

    generation_keychain = Keychain.init(5, sharding_config=replicated_language_model.sharding_config)
    max_output_length = 5  # very short cuz precision issues are insane
    padded_input_length = next(length for length in _COMPILED_PROMPT_LENGTHS if length >= token_ids.size)
    padded_token_ids = jnp.pad(token_ids, (0, padded_input_length - token_ids.size))
    eager_token_ids, eager_lengths = _sharded_generation_batch(
        replicated_language_model,
        padded_token_ids[None, :],
        jnp.array([token_ids.size], dtype=jnp.int32),
    )
    eager_token_ids = replicated_language_model.generate_tokens(
        eager_token_ids,
        generation_config=generation_config,
        prompt_lengths_without_padding=eager_lengths,
        max_output_length=max_output_length,
        prefill_forward_pass_config=prefill_forward_pass_config,
        decode_forward_pass_config=decode_forward_pass_config,
        keychain=generation_keychain,
    ).token_ids
    eager_token_ids = _take_first_batch_row(replicated_language_model, eager_token_ids)

    streaming_token_generator = replicated_language_model.stream_tokens(
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
