import asyncio
import json
from dataclasses import replace
from typing import cast

import httpx2
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xgrammar as xg
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessage
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from lalamo.data.huggingface_message import HFConversation
from lalamo.inference.batch_scheduler import BatchSchedulerConfig, ContinuousBatchScheduler, FixedSizeBatchScheduler
from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    GeneratedToken,
    SequenceFinished,
    TokenEvent,
)
from lalamo.initializer import RandomInitializer
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig, ToolCallFormat, UserMessage
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules.activations import SiLU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.mamba import Mamba2Config
from lalamo.server import create_app
from lalamo.utils.json import JSON
from tests.helpers import build_tiny_attention_decoder_config, dense_log_softmax_rows


@pytest.fixture(scope="module")
def recurrent_model() -> LanguageModel:
    decoder = build_tiny_attention_decoder_config((None,))
    (layer,) = decoder.transformer_config.layer_configs
    mixer = Mamba2Config(
        in_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        conv_config=SeparableCausalConvConfig(has_biases=False),
        activation=SiLU(),
        kernel_size=3,
        num_heads=2,
        num_groups=1,
        head_dim=2,
        state_dim=3,
        has_in_biases=False,
        has_out_biases=False,
    )
    decoder = replace(
        decoder,
        transformer_config=replace(
            decoder.transformer_config,
            layer_configs=(replace(layer, mixer_config=mixer, rope_config=None),),
        ),
    )
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig(
            prompt_template="",
            output_parser_regex=None,
            system_role_name="system",
            user_role_name="user",
            assistant_role_name="assistant",
            eos_token=None,
            bos_token=None,
        ),
        decoder_config=decoder,
        generation_config=GenerationConfig(),
    )
    return config.init(
        Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:1]),
            key=jax.random.key(7),
        ),
    )


@pytest.mark.fast
def test_app_lifespan_reentry_serves_real_http_requests(recurrent_model: LanguageModel) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "prompt": 1,
                **{f"token{index}": index for index in range(2, recurrent_model.decoder.vocab_size)},
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.decoder = Fuse()
    codec = replace(recurrent_model.token_codec.config, prompt_template="prompt").init(tokenizer)
    model = replace(
        recurrent_model, token_codec=codec, config=replace(recurrent_model.config, token_codec_config=codec.config)
    )
    api = create_app(
        model, "test", ContinuousBatchingConfig(slot_count=1, max_context_length=16, prefill_batch_size=1)
    )

    async def complete() -> httpx2.Response:
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=api), base_url="http://local") as http:
            return await asyncio.wait_for(
                http.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "prompt"}],
                        "temperature": 0,
                        "logit_bias": {"3": 100},
                        "max_completion_tokens": 1,
                    },
                ),
                timeout=10,
            )

    for _ in range(2):
        with TestClient(api) as http:
            assert http.get("/health").status_code == 200
            response = asyncio.run(complete())
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "token3"
            assert response.json()["usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


@pytest.mark.fast
def test_recurrent_requests_match_dense_generation_after_slot_and_prefix_reuse(recurrent_model: LanguageModel) -> None:
    engine = ContinuousBatchingEngine(
        recurrent_model,
        ContinuousBatchingConfig(slot_count=2, max_context_length=48, prefill_batch_size=2, prefill_chunk_size=3),
    )
    first = tuple(range(1, 24))
    prompts = [first, tuple(range(3, 21)), (*first[:15], 30)]
    budgets = [5, 9, 6]
    events: list[list[TokenEvent]] = [[], [], []]
    generation = GenerationConfig(temperature=0.0)
    for prompt, budget, received in zip(prompts[:2], budgets[:2], events[:2], strict=True):
        engine.submit(prompt, budget, generation, 0, return_logprobs=True, on_events=received.extend)
    assert engine.step()
    engine.submit(prompts[2], budgets[2], generation, 0, return_logprobs=True, on_events=events[2].extend)
    for _ in range(20):
        if not engine.step():
            break
    assert not engine.step()
    for prompt, budget, received in zip(prompts, budgets, events, strict=True):
        assert received[-1] == SequenceFinished(FinishReason.LENGTH, budget)
        expected = recurrent_model.stream_tokens(
            jnp.asarray(prompt, dtype=jnp.int32),
            generation,
            budget,
            keychain=Keychain.init(0, sharding_config=recurrent_model.sharding_config),
        )
        token_ids = [event.token_id for event in received if isinstance(event, GeneratedToken)]
        assert token_ids == list(map(int, expected))
        rows = dense_log_softmax_rows(recurrent_model, prompt, token_ids)
        for event, row in zip(received[:-1], rows[:-1], strict=True):
            assert isinstance(event, GeneratedToken)
            assert event.logprobs is not None
            np.testing.assert_allclose(event.logprobs.logprob, row[event.token_id], rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(
                event.logprobs.top_logprobs,
                row[jnp.asarray(event.logprobs.top_token_ids)],
                rtol=1e-4,
                atol=1e-3,
            )


@pytest.mark.fast
def test_recurrent_auto_admission_uses_fp32_state_and_padded_prefill(
    recurrent_model: LanguageModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One row has 20 conv and 12 SSM fp32 values, even though inference activations use bf16.
    # 11200 available bytes admit two rows. A third would fit if state were miscounted as bf16 or prefill unpadded.
    monkeypatch.setattr("lalamo.inference.continuous_batching.get_free_bytes", lambda _device: 11200)
    config = ContinuousBatchingConfig(max_context_length=48, prefill_batch_size=3, prefill_chunk_size=3)
    engine = ContinuousBatchingEngine(recurrent_model, config)
    assert engine.slot_count == 2
    with pytest.raises(ValueError, match="Not enough device memory"):
        monkeypatch.setattr("lalamo.inference.continuous_batching.get_free_bytes", lambda _device: 1000)
        ContinuousBatchingEngine(recurrent_model, config)


@pytest.mark.fast
def test_recurrent_context_requires_explicit_limit(recurrent_model: LanguageModel) -> None:
    with pytest.raises(ValueError, match="requires max_context_length"):
        ContinuousBatchingEngine(recurrent_model, ContinuousBatchingConfig(slot_count=1))


@pytest.mark.fast
def test_logit_bias_applies_to_continuous_and_dense_generation(recurrent_model: LanguageModel) -> None:
    engine = ContinuousBatchingEngine(
        recurrent_model,
        ContinuousBatchingConfig(slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=4),
    )
    prompt = (1, 2, 3, 4)
    generations = (
        GenerationConfig(temperature=0.0, logit_bias=((7, 100.0),)),
        GenerationConfig(
            temperature=2.0,
            top_k=1,
            top_p=0.1,
            min_p=0.5,
            repetition_penalty=1.2,
            presence_penalty=1.0,
            frequency_penalty=1.0,
            logit_bias=((9, 100.0), (7, -100.0)),
        ),
    )
    events: list[list[TokenEvent]] = [[], []]
    for generation, received in zip(generations, events, strict=True):
        engine.submit(prompt, 4, generation, 0, return_logprobs=True, on_events=received.extend)
    while engine.step():
        pass
    for generation, received, expected in zip(generations, events, (7, 9), strict=True):
        token_ids = [event.token_id for event in received if isinstance(event, GeneratedToken)]
        assert token_ids == [expected] * 4
        dense = recurrent_model.stream_tokens(
            jnp.asarray(prompt, dtype=jnp.int32),
            generation,
            4,
            keychain=Keychain.init(0, sharding_config=recurrent_model.sharding_config),
        )
        assert list(map(int, dense)) == token_ids
        rows = dense_log_softmax_rows(recurrent_model, prompt, token_ids)
        for event, row in zip(received[:-1], rows[:-1], strict=True):
            assert isinstance(event, GeneratedToken)
            assert event.logprobs is not None
            np.testing.assert_allclose(event.logprobs.logprob, row[event.token_id], rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(
                event.logprobs.top_logprobs,
                row[jnp.asarray(event.logprobs.top_token_ids)],
                rtol=1e-4,
                atol=1e-3,
            )


@pytest.mark.fast
@pytest.mark.parametrize(
    "generation",
    [
        GenerationConfig(temperature=0.0, presence_penalty=1.0),
        GenerationConfig(temperature=0.0, presence_penalty=-1.0),
        GenerationConfig(temperature=0.0, frequency_penalty=1.0),
        GenerationConfig(temperature=0.0, frequency_penalty=-1.0),
    ],
)
def test_additive_penalties_use_generated_history_in_continuous_and_dense_generation(
    recurrent_model: LanguageModel, generation: GenerationConfig
) -> None:
    prompt = (1, 2, 3, 4)
    (initial_row,) = dense_log_softmax_rows(recurrent_model, prompt, [])
    desired = np.full(recurrent_model.decoder.vocab_size, -20.0, dtype=np.float32)
    desired[1], desired[7] = 0.5, 0.0
    biases = desired - initial_row
    generation = replace(generation, logit_bias=tuple(enumerate(map(float, biases))))
    counts = np.zeros(recurrent_model.decoder.vocab_size, dtype=np.int32)
    expected = []
    for _ in range(4):
        row = dense_log_softmax_rows(recurrent_model, prompt, expected)[-1]
        penalized = row + biases - (generation.presence_penalty or 0.0) * (counts > 0)
        penalized -= (generation.frequency_penalty or 0.0) * counts
        token = int(np.argmax(penalized))
        expected.append(token)
        counts[token] += 1
    assert expected[0] == 1

    engine = ContinuousBatchingEngine(
        recurrent_model,
        ContinuousBatchingConfig(slot_count=1, max_context_length=32, prefill_chunk_size=4),
    )
    events: list[TokenEvent] = []
    engine.submit(prompt, 4, generation, 0, return_logprobs=True, on_events=events.extend)
    while engine.step():
        pass
    generated = [event for event in events if isinstance(event, GeneratedToken)]
    assert [event.token_id for event in generated] == expected
    keychain = Keychain.init(0, sharding_config=recurrent_model.sharding_config)
    streamed = recurrent_model.stream_tokens(jnp.asarray(prompt, dtype=jnp.int32), generation, 4, keychain=keychain)
    assert list(map(int, streamed)) == expected
    dense = recurrent_model.generate_tokens(
        jnp.asarray([prompt], dtype=jnp.int32), generation, max_output_length=4, keychain=keychain
    )
    assert list(map(int, dense.token_ids[0])) == expected
    for event, row in zip(generated, dense_log_softmax_rows(recurrent_model, prompt, expected)[:-1], strict=True):
        assert event.logprobs is not None
        np.testing.assert_allclose(event.logprobs.logprob, row[event.token_id], rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(
            event.logprobs.top_logprobs, row[jnp.asarray(event.logprobs.top_token_ids)], rtol=1e-4, atol=1e-3
        )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_stop", "generation_stop", "eos", "expected_count"),
    [(8, (7,), None, 1), (7, (), None, 4), (7, None, None, 1), (8, (7,), (8,), 4), (8, (), (7,), 1)],
)
def test_dense_and_streaming_stop_configuration_precedence(
    recurrent_model: LanguageModel,
    model_stop: int,
    generation_stop: tuple[int, ...] | None,
    eos: tuple[int, ...] | None,
    expected_count: int,
) -> None:
    generation = GenerationConfig(temperature=0.0, stop_token_ids=(model_stop,), logit_bias=((7, 100.0),))
    model = replace(recurrent_model, config=replace(recurrent_model.config, generation_config=generation))
    override = None
    if generation_stop is not None:
        override = replace(generation, stop_token_ids=generation_stop)
    stop_ids = None
    if eos is not None:
        stop_ids = jnp.asarray(eos, dtype=jnp.int32)
    keychain = Keychain.init(0, sharding_config=model.sharding_config)
    prompt = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
    streamed = model.stream_tokens(prompt, override, 4, eos_token_ids=stop_ids, keychain=keychain)
    assert list(map(int, streamed)) == [7] * expected_count
    dense = model.generate_tokens(
        prompt[None, :], override, max_output_length=4, eos_token_ids=stop_ids, keychain=keychain
    )
    assert dense.token_ids[0].tolist() == [7] * expected_count + [0] * (4 - expected_count)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        (None, "prefix"),
        (GenerationConfig(stop_token_ids=(), temperature=0.0, logit_bias=((2, 100.0),)), "prefix" * 4),
        (GenerationConfig(stop_token_ids=(7,), temperature=0.0, logit_bias=((7, 100.0),)), "done"),
        (
            GenerationConfig(
                stop_token_ids=(7,), temperature=0.0, presence_penalty=2.0, logit_bias=((2, 100.0), (7, 99.0))
            ),
            "prefixdone",
        ),
    ],
)
def test_reply_stop_configuration_matches_fixed_and_continuous_batches(
    recurrent_model: LanguageModel, generation: GenerationConfig | None, expected: str
) -> None:
    vocabulary = {
        "padding": 0,
        "prompt": 1,
        "prefix": 2,
        "done": 7,
        "[UNK]": recurrent_model.decoder.vocab_size - 1,
        **{f"token{index}": index for index in range(3, recurrent_model.decoder.vocab_size - 1) if index != 7},
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()
    codec = replace(recurrent_model.token_codec.config, prompt_template="prompt").init(tokenizer)
    defaults = GenerationConfig(stop_token_ids=(2,), temperature=0.0, logit_bias=((2, 100.0),))
    model = replace(
        recurrent_model,
        token_codec=codec,
        config=replace(recurrent_model.config, token_codec_config=codec.config, generation_config=defaults),
    )
    messages = (UserMessage("please reply"),)
    response = model.reply(
        messages, generation, max_output_length=4, keychain=Keychain.init(0, sharding_config=model.sharding_config)
    )
    assert response.response == expected
    conversations = [HFConversation(messages, None)] * 3
    for scheduler in (FixedSizeBatchScheduler(model), ContinuousBatchScheduler(model)):
        replies = scheduler.reply_many(
            conversations, generation, BatchSchedulerConfig(batch_size=2, max_output_length=4, padded_length=4)
        )
        assert {index: reply.response for index, reply in replies} == dict.fromkeys(range(3), expected)


@pytest.mark.fast
def test_signed_64_bit_seeds_replay_after_slot_reuse(recurrent_model: LanguageModel) -> None:
    engine = ContinuousBatchingEngine(
        recurrent_model,
        ContinuousBatchingConfig(slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=4),
    )
    prompt = (1, 2, 3, 4)
    generation = GenerationConfig(temperature=1.0)
    seeds = (0, 2**32, -1, -(2**63), 2**63 - 1)
    original: list[list[TokenEvent]] = [[] for _ in seeds]
    for seed, received in zip(seeds, original, strict=True):
        engine.submit(prompt, 8, generation, seed, on_events=received.extend)
    while engine.step():
        pass
    replayed: list[list[TokenEvent]] = [[] for _ in seeds]
    for seed, received in reversed(tuple(zip(seeds, replayed, strict=True))):
        engine.submit(prompt, 8, generation, seed, on_events=received.extend)
    while engine.step():
        pass
    sequences = [tuple(event.token_id for event in events if isinstance(event, GeneratedToken)) for events in original]
    assert len(set(sequences)) == len(seeds)
    for first, second in zip(original, replayed, strict=True):
        assert first == second


@pytest.mark.fast
@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize(
    "name", ["total_pages", "slot_count", "max_context_length", "prefill_batch_size", "prefill_chunk_size"]
)
def test_nonpositive_capacity_is_rejected(name: str, value: int) -> None:
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        replace(ContinuousBatchingConfig(), **{name: value})


@pytest.fixture(params=list(ToolCallFormat))
def tool_choice_model(recurrent_model: LanguageModel, request: pytest.FixtureRequest) -> LanguageModel:
    tool_format = request.param
    match tool_format:
        case ToolCallFormat.QWEN_XML:
            beginning = "<tool_call>\n<function="
            wanted = "wanted"
            other = "other"
            ending = ">\n</function>\n</tool_call>"
        case ToolCallFormat.LIQUID:
            beginning = "<|tool_call_start|>["
            wanted = "wanted"
            other = "other"
            ending = "()]<|tool_call_end|>"
        case ToolCallFormat.MUSE_ATEM:
            beginning = " to="
            wanted = 'wanted<|message|><atem:function_calls><atem:invoke name="wanted"'
            other = 'other<|message|><atem:function_calls><atem:invoke name="other"'
            ending = "></atem:invoke></atem:function_calls><|eom|>"
        case _:
            raise AssertionError(tool_format)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "prompt": 1,
                "<eos>": 2,
                "plain answer": 3,
                beginning + wanted + ending: 4,
                beginning + other + ending: 5,
                (beginning + other + ending).replace("other", "unlisted"): 6,
                beginning: 7,
                wanted: 8,
                other: 9,
                ending: 10,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.decoder = Fuse()
    codec = replace(
        recurrent_model.token_codec.config,
        prompt_template="prompt",
        tool_call_format=tool_format,
        eos_token="<eos>",
    ).init(tokenizer)
    return LanguageModel(
        config=replace(
            recurrent_model.config,
            token_codec_config=codec.config,
            generation_config=GenerationConfig(stop_token_ids=(2,)),
        ),
        token_codec=codec,
        decoder=recurrent_model.decoder,
        sharding_config=recurrent_model.sharding_config,
    )


def _tool_matcher(model: LanguageModel, *, named: bool) -> xg.GrammarMatcher:
    codec = model.token_codec
    info = xg.TokenizerInfo(
        [codec.decode_token_bytes(token_id) for token_id in range(codec.tokenizer.get_vocab_size())],
        vocab_size=model.decoder.vocab_size,
        stop_token_ids=[2],
    )
    choices = [codec.decode_tokens([4])]
    if not named:
        choices.append(codec.decode_tokens([5]))
    grammar = "root ::= " + " | ".join(json.dumps(choice) for choice in choices)
    return xg.GrammarMatcher(xg.GrammarCompiler(info).compile_grammar(grammar))


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
def test_unsatisfiable_tool_constraint_fails_request_and_keeps_worker_healthy(
    recurrent_model: LanguageModel, stream: bool
) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "prompt": 1, "<eos>": 2, "<tool_call>\n<function=broken>\n": 3, "safe": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.decoder = Fuse()
    codec = replace(
        recurrent_model.token_codec.config,
        prompt_template="prompt",
        tool_call_format=ToolCallFormat.QWEN_XML,
        eos_token="<eos>",
    ).init(tokenizer)
    model = LanguageModel(
        config=replace(
            recurrent_model.config,
            token_codec_config=codec.config,
            generation_config=GenerationConfig(stop_token_ids=(2,)),
        ),
        token_codec=codec,
        decoder=recurrent_model.decoder,
        sharding_config=recurrent_model.sharding_config,
    )
    config = ContinuousBatchingConfig(slot_count=2, max_context_length=16, prefill_batch_size=2, prefill_chunk_size=3)
    with TestClient(create_app(model, "test", config), raise_server_exceptions=False) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "prompt"}],
                "tools": [{"type": "function", "function": {"name": "broken"}}],
                "tool_choice": "required",
                "logit_bias": {"3": 100},
                "max_completion_tokens": 4,
                "n": 2,
                "stream": stream,
            },
        )
        if stream:
            assert response.status_code == 200
            packets = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            error = packets[-1]["error"]
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            assert response.status_code == 400
            error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "no valid next token" in error["message"]
        assert http.get("/health").status_code == 200
        normal = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "prompt"}],
                "logit_bias": {"4": 100},
                "max_completion_tokens": 2,
                "n": 2,
            },
        )
        assert normal.status_code == 200
        assert [choice["message"]["content"] for choice in normal.json()["choices"]] == ["safesafe"] * 2


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["json", "json_with_tools", "tool", "required", "length", "stop"])
@pytest.mark.parametrize("format_type", ["json_object", "json_schema"])
def test_json_mode_masks_invalid_content_and_preserves_real_tool_choices(
    tool_choice_model: LanguageModel, stream: bool, mode: str, format_type: str
) -> None:
    native_tool = tool_choice_model.token_codec.decode_tokens([4])
    content = json.dumps({"quoted": native_tool, "markers": "</think><|eom|><|eot|>", "value": "café"})
    native_json = content
    partial = '{"value":"'
    stopped = json.dumps({"value": native_tool + "CUTafter"})
    native_stopped = stopped
    invalid_content = "invalid plaintext"
    if format_type == "json_schema":
        invalid_content = json.dumps({"quoted": native_tool, "markers": "wrong", "value": "café"})
    if tool_choice_model.token_codec.config.tool_call_format is ToolCallFormat.MUSE_ATEM:
        native_json = "to=user<|message|>" + content
        partial = "to=user<|message|>" + partial
        native_stopped = "to=user<|message|>" + stopped
        if format_type == "json_schema":
            invalid_content = "to=user<|message|>" + invalid_content
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "prompt": 1,
                "<eos>": 2,
                invalid_content: 3,
                native_json: 4,
                native_tool: 5,
                partial: 6,
                native_stopped: 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.decoder = Fuse()
    codec = tool_choice_model.token_codec.config.init(tokenizer)
    model = LanguageModel(
        config=tool_choice_model.config,
        token_codec=codec,
        decoder=tool_choice_model.decoder,
        sharding_config=tool_choice_model.sharding_config,
    )
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "Reply in JSON."}],
        "response_format": {"type": "json_object"},
        "logit_bias": {"2": 100, "3": 95, "4": 90, "5": -100},
        "max_completion_tokens": 4,
        "n": 2,
        "stream": stream,
        "logprobs": True,
        "top_logprobs": 20,
    }
    if format_type == "json_schema":
        body["messages"] = [{"role": "user", "content": "Reply."}]
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "description": "A result containing quoted native controls.",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "quoted": {"type": "string"},
                        "markers": {"type": "string", "enum": ["</think><|eom|><|eot|>"]},
                        "value": {"type": "string"},
                    },
                    "required": ["quoted", "markers", "value"],
                    "additionalProperties": False,
                },
            },
        }
    if mode != "json":
        body["tools"] = [{"type": "function", "function": {"name": "wanted"}}]
        body["parallel_tool_calls"] = False
    if mode in ("tool", "required"):
        body["logit_bias"] = {"2": 100, "3": 95, "4": 80, "5": 90}
    if mode == "required":
        body["tool_choice"] = "required"
        body["logit_bias"] = {"2": 100, "3": 95, "4": 99, "5": 90}
    expected_content = content
    expected_finish = "stop"
    if mode == "length":
        body["max_completion_tokens"] = 1
        body["logit_bias"] = {"2": 100, "3": 95, "6": 99}
        expected_content = '{"value":"'
        expected_finish = "length"
    elif mode == "stop":
        body["stop"] = "CUT"
        body["logit_bias"] = {"2": 100, "3": 95, "7": 99}
        expected_content = stopped.split("CUT")[0]
    if format_type == "json_schema" and mode in ("length", "stop"):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "partial_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    config = ContinuousBatchingConfig(slot_count=2, max_context_length=16, prefill_batch_size=2, prefill_chunk_size=3)
    with TestClient(create_app(model, "test", config)) as http:
        response = http.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        if stream:
            packets = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert response.text.endswith("data: [DONE]\n\n")
            choices = [choice for packet in packets for choice in packet["choices"]]
            message_field = "delta"
        else:
            choices = response.json()["choices"]
            message_field = "message"
        for index in range(2):
            row = [choice for choice in choices if choice["index"] == index]
            messages = [choice[message_field] for choice in row]
            text = "".join(message.get("content") or "" for message in messages)
            calls = [call for message in messages for call in message.get("tool_calls", [])]
            if mode in ("tool", "required"):
                assert text == "" and len(calls) == 1
                assert calls[0]["function"] == {"name": "wanted", "arguments": "{}"}
                assert row[-1]["finish_reason"] == "tool_calls"
            else:
                assert text == expected_content and not calls
                if mode not in ("length", "stop"):
                    assert json.loads(text)["quoted"] == native_tool
                assert row[-1]["finish_reason"] == expected_finish
        invalid_request = {**body, "messages": [{"role": "user", "content": "Reply."}], "stream": False}
        if format_type == "json_schema":
            invalid_request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "bad", "schema": {"type": "array", "uniqueItems": True}},
            }
        invalid_response = http.post("/v1/chat/completions", json=invalid_request)
        assert invalid_response.status_code == 400
        assert invalid_response.json()["error"]["param"].startswith(
            "response_format" if format_type == "json_schema" else "messages"
        )
        normal = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Reply."}],
                "logit_bias": {"3": 100},
                "max_completion_tokens": 2,
            },
        )
        assert normal.status_code == 200
        assert normal.json()["choices"][0]["message"]["content"] == invalid_content * 2
        assert http.get("/health").status_code == 200


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
def test_json_schema_cache_collision_rejects_before_forced_or_public_generation(
    tool_choice_model: LanguageModel,
    stream: bool,
) -> None:
    schema: dict[str, JSON] = {
        "type": "object",
        "properties": {
            name: {
                "type": "object",
                "properties": {"title": {"type": field_type}},
                "required": ["title"],
                "additionalProperties": False,
            }
            for name, field_type in (("a", "string"), ("b", "integer"))
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    config = ContinuousBatchingConfig(slot_count=2, max_context_length=16, prefill_batch_size=2, prefill_chunk_size=3)
    with TestClient(create_app(tool_choice_model, "test", config)) as http:
        for tool_choice in ("auto", "required", {"type": "function", "function": {"name": "wanted"}}):
            response = http.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Reply."}],
                    "tools": [{"type": "function", "function": {"name": "wanted"}}],
                    "tool_choice": tool_choice,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "result", "schema": schema, "strict": True},
                    },
                    "stream": stream,
                    "n": 2,
                    "max_completion_tokens": 4,
                },
            )
            assert response.status_code == 400
            error = response.json()["error"]
            assert error["type"] == "invalid_request_error"
            assert error["param"].startswith("response_format")
            assert "parser cache" in error["message"]
        normal = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Reply."}],
                "logit_bias": {"3": 100},
                "max_completion_tokens": 2,
                "n": 2,
            },
        )
        assert normal.status_code == 200
        assert [choice["message"]["content"] for choice in normal.json()["choices"]] == ["plain answer" * 2] * 2


@pytest.mark.fast
def test_strict_wire_validation_preserves_ordinary_compositions_and_inactive_tools(
    tool_choice_model: LanguageModel,
) -> None:
    composed: dict[str, JSON] = {
        "allOf": [
            {"type": "object", "properties": {"count": {"type": "integer"}}},
            {"properties": {"count": {"minimum": 1}}},
        ]
    }
    closed: dict[str, JSON] = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": False,
    }
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "Reply."}],
        "logit_bias": {"3": 100},
        "max_completion_tokens": 2,
        "n": 2,
    }
    inactive_choices: tuple[JSON, ...] = (
        "none",
        {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}},
    )
    invalid_schemas: tuple[dict[str, JSON], ...] = (
        {},
        composed,
        {"type": "object", "properties": {"count": {"type": "integer"}}, "additionalProperties": False},
        {**closed, "properties": {"count": {"type": "integer", "multipleOf": 2}}},
        {**closed, "properties": {"count": {"type": "integer", "enum": [1, 3], "minimum": 2}}},
        {**closed, "properties": {"count": {"type": "array", "items": {"type": "integer"}, "const": [2**64 + 1]}}},
        {
            **closed,
            "properties": {
                "count": {
                    "type": "object",
                    "properties": {"child": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["child"],
                    "additionalProperties": False,
                    "enum": [{"child": [2**64 + 1]}],
                }
            },
        },
    )
    config = ContinuousBatchingConfig(slot_count=2, max_context_length=16, prefill_batch_size=2, prefill_chunk_size=3)
    with TestClient(create_app(tool_choice_model, "test", config)) as http:
        for strict in (False, None):
            response = http.post(
                "/v1/chat/completions",
                json={
                    **body,
                    "tools": [
                        {"type": "function", "function": {"name": "wanted", "parameters": composed, "strict": strict}}
                    ],
                },
            )
            assert response.status_code == 200
            assert [choice["message"]["content"] for choice in response.json()["choices"]] == ["plain answer" * 2] * 2
        for tool_choice in inactive_choices:
            for parameters in (None, closed):
                response = http.post(
                    "/v1/chat/completions",
                    json={
                        **body,
                        "tools": [
                            {
                                "type": "function",
                                "function": {"name": "wanted", "parameters": parameters, "strict": True},
                            }
                        ],
                        "tool_choice": tool_choice,
                    },
                )
                assert response.status_code == 200
                assert [choice["message"]["content"] for choice in response.json()["choices"]] == [
                    "plain answer" * 2
                ] * 2
            for parameters in invalid_schemas:
                response = http.post(
                    "/v1/chat/completions",
                    json={
                        **body,
                        "tools": [
                            {
                                "type": "function",
                                "function": {"name": "wanted", "parameters": parameters, "strict": True},
                            }
                        ],
                        "tool_choice": tool_choice,
                    },
                )
                assert response.status_code == 400
                assert response.json()["error"]["type"] == "invalid_request_error"
                assert response.json()["error"]["param"].startswith("tools.0.function")
        healthy = http.post("/v1/chat/completions", json=body)
        assert healthy.status_code == 200
        assert [choice["message"]["content"] for choice in healthy.json()["choices"]] == ["plain answer" * 2] * 2


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["auto_call", "required", "named", "ordinary", "zeroargs", "text", "json"])
def test_strict_tools_constrain_real_http_sampling_without_changing_raw_logprobs(
    tool_choice_model: LanguageModel,
    stream: bool,
    mode: str,
) -> None:
    empty_call = tool_choice_model.token_codec.decode_tokens([4])
    tool_format = tool_choice_model.token_codec.config.tool_call_format
    match tool_format:
        case ToolCallFormat.QWEN_XML:
            valid_call = empty_call.replace("</function>", "<parameter=count>2</parameter>\n</function>")
        case ToolCallFormat.LIQUID:
            valid_call = empty_call.replace("wanted()", "wanted(count=2)")
        case ToolCallFormat.MUSE_ATEM:
            valid_call = empty_call.replace(
                "</atem:invoke>", '<atem:parameter name="count">2</atem:parameter></atem:invoke>'
            )
        case _:
            raise AssertionError(tool_format)
    invalid_call = valid_call.replace(">2<", ">1<").replace("count=2)", "count=1)")
    ordinary_call = invalid_call.replace("wanted", "other")
    text = "plain answer"
    good_json, bad_json = '{"value":"ok"}', '{"value":"bad"}'
    if tool_format is ToolCallFormat.MUSE_ATEM:
        text = "to=user<|message|>" + text + "<|eot|>"
        good_json = "to=user<|message|>" + good_json
        bad_json = "to=user<|message|>" + bad_json
    vocabulary = {
        "[UNK]": 0,
        "prompt": 1,
        "<eos>": 2,
        text: 3,
        valid_call: 4,
        invalid_call: 5,
        empty_call: 6,
        ordinary_call: 7,
        good_json: 8,
        bad_json: 9,
        **{f"unused{index}": index for index in range(10, tool_choice_model.decoder.vocab_size)},
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()
    codec = tool_choice_model.token_codec.config.init(tokenizer)
    model = LanguageModel(
        config=replace(tool_choice_model.config, token_codec_config=codec.config),
        token_codec=codec,
        decoder=tool_choice_model.decoder,
        sharding_config=tool_choice_model.sharding_config,
    )
    parameters: dict[str, JSON] = {
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 2, "maximum": 2}},
        "required": ["count"],
        "additionalProperties": False,
    }
    tools: list[JSON] = [
        {"type": "function", "function": {"name": "wanted", "strict": True, "parameters": parameters}}
    ]
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "Reply in JSON."}],
        "tools": tools,
        "parallel_tool_calls": False,
        "temperature": 0.7,
        "top_k": 1,
        "top_p": 0.1,
        "min_p": 0.8,
        "repetition_penalty": 2.0,
        "presence_penalty": 1.0,
        "frequency_penalty": -1.0,
        "seed": 0,
        "logit_bias": {"2": -100, "3": -100, "5": 100, "6": 99, "7": 98, "4": 90},
        "max_completion_tokens": 4,
        "logprobs": True,
        "top_logprobs": 20,
        "n": 2,
        "stream": stream,
    }
    expected_function, expected_arguments = "wanted", {"count": 2}
    if mode == "required":
        body["tool_choice"] = "required"
        body["logit_bias"] = {"2": 100, "3": 99, "5": 98, "6": 97, "7": 96, "4": 90}
    elif mode in ("named", "ordinary", "zeroargs"):
        selected_name = "wanted"
        if mode in ("named", "ordinary"):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "other",
                        "strict": False,
                        "parameters": {
                            "allOf": [{"type": "object", "properties": {"count": {"type": "integer", "minimum": 3}}}]
                        },
                    },
                }
            )
        if mode == "ordinary":
            selected_name = expected_function = "other"
            expected_arguments = {"count": 1}
            body["logit_bias"] = {"2": 100, "3": 99, "5": 98, "6": 97, "4": 96, "7": 90}
        elif mode == "zeroargs":
            tools[0] = {"type": "function", "function": {"name": "wanted", "strict": True, "parameters": None}}
            expected_arguments = {}
            body["logit_bias"] = {"2": 100, "3": 99, "5": 98, "4": 97, "7": 96, "6": 90}
        else:
            body["logit_bias"] = {"2": 100, "3": 99, "5": 98, "6": 97, "7": 96, "4": 90}
        body["tool_choice"] = {"type": "function", "function": {"name": selected_name}}
    elif mode == "text":
        body["max_completion_tokens"] = 1
        body["logit_bias"] = {"2": 90, "3": 100, "5": 99, "6": 98, "7": 97, "4": 70}
    elif mode == "json":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "Result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "enum": ["ok"]}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        body["logit_bias"] = {"2": 100, "3": 99, "5": 98, "6": 97, "7": 96, "9": 95, "8": 90, "4": 70}
    config = ContinuousBatchingConfig(slot_count=2, max_context_length=16, prefill_batch_size=2, prefill_chunk_size=3)
    with TestClient(create_app(model, "test", config)) as http:
        response = http.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        if stream:
            packets = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            choices = [choice for packet in packets for choice in packet["choices"]]
            message_field = "delta"
        else:
            choices = response.json()["choices"]
            message_field = "message"
        for index in range(2):
            row = [choice for choice in choices if choice["index"] == index]
            messages = [choice[message_field] for choice in row]
            content = "".join(message.get("content") or "" for message in messages)
            calls = [call for message in messages for call in message.get("tool_calls", [])]
            entries = [entry for choice in row for entry in (choice.get("logprobs") or {}).get("content", [])]
            if mode in ("text", "json"):
                assert content == ("plain answer" if mode == "text" else '{"value":"ok"}')
                assert not calls
                assert row[-1]["finish_reason"] == ("length" if mode == "text" else "stop")
                raw = dense_log_softmax_rows(model, (1,), [])[0]
                (entry,) = entries
                sampled = 3 if mode == "text" else 8
                expected_top = set(map(int, np.argsort(-raw)[:20]))
                assert entry["bytes"] == list(codec.decode_token_bytes(sampled))
                np.testing.assert_allclose(
                    entry["logprob"], raw[sampled] if sampled in expected_top else -9999, atol=1e-3
                )
                by_bytes = {tuple(codec.decode_token_bytes(token)): token for token in range(model.decoder.vocab_size)}
                assert {by_bytes[tuple(item["bytes"])] for item in entry["top_logprobs"]} == expected_top
                for item in entry["top_logprobs"]:
                    np.testing.assert_allclose(item["logprob"], raw[by_bytes[tuple(item["bytes"])]], atol=1e-3)
            else:
                assert not content and not entries
                (call,) = calls
                assert call["function"]["name"] == expected_function
                assert json.loads(call["function"]["arguments"]) == expected_arguments
                assert row[-1]["finish_reason"] == "tool_calls"
        healthy = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Reply."}],
                "logit_bias": {"10": 100},
                "max_completion_tokens": 1,
            },
        )
        assert healthy.status_code == 200
        assert healthy.json()["choices"][0]["message"]["content"] == "unused10"


@pytest.mark.fast
@pytest.mark.parametrize("named", [False, True], ids=["required", "named"])
@pytest.mark.parametrize("temperature", [0.0, 0.7], ids=["greedy", "filtered"])
def test_tool_choice_masks_eos_plaintext_and_wrong_names_before_sampling(
    tool_choice_model: LanguageModel, named: bool, temperature: float
) -> None:
    model = tool_choice_model
    engine = ContinuousBatchingEngine(
        model,
        ContinuousBatchingConfig(slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=3),
    )
    prompt = (1, 1, 1, 1, 1)
    generation = GenerationConfig(
        temperature=temperature,
        top_k=1,
        top_p=0.1,
        min_p=0.9,
        stop_token_ids=(2,),
        logit_bias=((2, 100.0), (3, 90.0), (6, 80.0), (9, 70.0), (7, 60.0), (8, 50.0), (10, 40.0)),
    )
    received: list[TokenEvent] = []
    engine.submit(
        prompt,
        4,
        generation,
        0,
        return_logprobs=True,
        grammar_matcher=_tool_matcher(model, named=named),
        on_events=received.extend,
    )
    for _ in range(10):
        if not engine.step():
            break
    assert not engine.step()
    expected = [7, 8 if named else 9, 10]
    generated = [event for event in received if isinstance(event, GeneratedToken)]
    assert [event.token_id for event in generated] == expected
    assert received[-1] == SequenceFinished(FinishReason.STOP, 4)
    rows = dense_log_softmax_rows(model, prompt, expected)
    for event, row in zip(generated, rows[:-1], strict=True):
        assert event.logprobs is not None
        np.testing.assert_allclose(event.logprobs.logprob, row[event.token_id], rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(
            event.logprobs.top_logprobs,
            row[jnp.asarray(event.logprobs.top_token_ids)],
            rtol=1e-4,
            atol=1e-3,
        )


@pytest.mark.fast
def test_cancelled_tool_matcher_does_not_contaminate_same_prompt_reuse(tool_choice_model: LanguageModel) -> None:
    model = tool_choice_model
    engine = ContinuousBatchingEngine(
        model,
        ContinuousBatchingConfig(slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=3),
    )
    prompt = (1, 1, 1, 1, 1)
    generation = GenerationConfig(
        temperature=0.0,
        stop_token_ids=(2,),
        logit_bias=((2, 100.0), (3, 90.0), (9, 70.0), (7, 60.0), (8, 50.0), (10, 40.0)),
    )
    cancelled_events: list[TokenEvent] = []
    cancel = engine.submit(
        prompt,
        4,
        generation,
        0,
        grammar_matcher=_tool_matcher(model, named=True),
        on_events=cancelled_events.extend,
    )
    for _ in range(10):
        if cancelled_events:
            break
        assert engine.step()
    assert [event.token_id for event in cancelled_events if isinstance(event, GeneratedToken)] == [7]
    cancel.set()
    assert not engine.step()
    codec = model.token_codec
    info = xg.TokenizerInfo(
        [codec.decode_token_bytes(token_id) for token_id in range(codec.tokenizer.get_vocab_size())],
        vocab_size=model.decoder.vocab_size,
        stop_token_ids=[2],
    )
    other = xg.GrammarMatcher(
        xg.GrammarCompiler(info).compile_grammar("root ::= " + json.dumps(codec.decode_tokens([5])))
    )
    received: list[TokenEvent] = []
    engine.submit(prompt, 4, generation, 0, grammar_matcher=other, on_events=received.extend)
    for _ in range(10):
        if not engine.step():
            break
    assert not engine.step()
    assert [event.token_id for event in received if isinstance(event, GeneratedToken)] == [7, 9, 10]
    assert received[-1] == SequenceFinished(FinishReason.STOP, 4)
    unconstrained: list[TokenEvent] = []
    engine.submit(prompt, 4, generation, 0, on_events=unconstrained.extend)
    for _ in range(10):
        if not engine.step():
            break
    assert not engine.step()
    assert unconstrained == [SequenceFinished(FinishReason.STOP, 1)]


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("named", [False, True], ids=["required", "named"])
@pytest.mark.parametrize("max_tokens", [1, 4], ids=["cutoff", "complete"])
def test_forced_tools_use_real_generation_and_isolate_each_choice(
    tool_choice_model: LanguageModel, stream: bool, named: bool, max_tokens: int
) -> None:
    tools: list[JSON] = [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
        for name in ("wanted", "other")
    ]
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "wanted"}} if named else "required",
        "temperature": 0,
        "logit_bias": {"2": 100, "3": -100, "4": 80, "5": 90, "6": 95},
        "seed": 0,
        "n": 2,
        "max_completion_tokens": max_tokens,
        "stream": stream,
        "logprobs": True,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    expected = "wanted" if named else "other"
    expected_finish = "length" if max_tokens == 1 else "tool_calls"
    expected_tokens = 1 if named or max_tokens == 1 else 2
    config = ContinuousBatchingConfig(slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=3)
    with TestClient(create_app(tool_choice_model, "test", config)) as http:
        response = http.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        if stream:
            chunks = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            choices = [choice for chunk in chunks for choice in chunk["choices"]]
            assert chunks[-1]["choices"] == []
            usage = chunks[-1]["usage"]
            calls = []
            for index in range(2):
                row = [choice for choice in choices if choice["index"] == index]
                assert row[0]["delta"]["role"] == "assistant"
                assert row[-1]["finish_reason"] == expected_finish
                assert "".join(choice["delta"].get("content", "") for choice in row) == ""
                row_calls = [call for choice in row for call in choice["delta"].get("tool_calls", [])]
                assert len(row_calls) == 1 and row_calls[0]["index"] == 0
                assert all(not choice["logprobs"] or not choice["logprobs"]["content"] for choice in row)
                calls.extend(row_calls)
        else:
            payload = response.json()
            choices = payload["choices"]
            usage = payload["usage"]
            assert {choice["index"] for choice in choices} == {0, 1}
            assert all(choice["finish_reason"] == expected_finish for choice in choices)
            assert all(choice["message"]["content"] in (None, "") for choice in choices)
            assert all(choice["logprobs"]["content"] == [] for choice in choices)
            assert all(len(choice["message"]["tool_calls"]) == 1 for choice in choices)
            calls = [choice["message"]["tool_calls"][0] for choice in choices]
        assert len({call["id"] for call in calls}) == 2
        assert all(call["type"] == "function" for call in calls)
        assert all(call["function"] == {"name": expected, "arguments": "{}"} for call in calls)
        assert usage == {
            "prompt_tokens": 1,
            "completion_tokens": 2 * expected_tokens,
            "total_tokens": 1 + 2 * expected_tokens,
        }
        followup = {
            **body,
            "tool_choice": {"type": "function", "function": {"name": "other"}},
            "stream": False,
            "stream_options": None,
            "n": 1,
            "max_completion_tokens": 4,
        }
        other = http.post("/v1/chat/completions", json=followup)
        assert other.status_code == 200
        assert other.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "other"
        unconstrained = http.post(
            "/v1/chat/completions",
            json={**followup, "tool_choice": "auto"},
        )
        assert unconstrained.status_code == 200
        assert unconstrained.json()["choices"][0]["finish_reason"] == "stop"
        assert not unconstrained.json()["choices"][0]["message"].get("tool_calls")
        assert unconstrained.json()["usage"]["completion_tokens"] == 1


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
def test_required_tool_budget_exhaustion_does_not_invent_a_call(
    tool_choice_model: LanguageModel, stream: bool
) -> None:
    model = tool_choice_model
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": [{"type": "function", "function": {"name": "wanted", "parameters": {"type": "object"}}}],
        "tool_choice": "required",
        "temperature": 0,
        "logit_bias": {"2": 100, "3": 100, "7": 90},
        "seed": 0,
        "max_completion_tokens": 1,
        "stream": stream,
        "logprobs": True,
        "top_logprobs": 3,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    config = ContinuousBatchingConfig(slot_count=1, max_context_length=32, prefill_batch_size=1, prefill_chunk_size=3)
    with TestClient(create_app(model, "test", config)) as http:
        response = http.post("/v1/chat/completions", json=body)
        healthy = http.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": body["messages"], "logit_bias": {"3": 100}, "max_completion_tokens": 1},
        )
        assert healthy.status_code == 200
        assert healthy.json()["choices"][0]["message"]["content"] == "plain answer"
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        choices = [choice for chunk in chunks for choice in chunk["choices"]]
        assert choices[-1]["finish_reason"] == "length"
        assert not any(choice["delta"].get("tool_calls") for choice in choices)
        content = "".join(choice["delta"].get("content", "") for choice in choices)
        entries = [entry for choice in choices if choice["logprobs"] for entry in choice["logprobs"]["content"]]
        usage = chunks[-1]["usage"]
    else:
        payload = response.json()
        (choice,) = payload["choices"]
        assert choice["finish_reason"] == "length"
        assert not choice["message"].get("tool_calls")
        content = choice["message"]["content"]
        entries = choice["logprobs"]["content"]
        usage = payload["usage"]
    assert not content and not entries
    assert usage == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


@pytest.mark.fast
def test_invalid_forced_choices_leave_the_real_api_healthy(tool_choice_model: LanguageModel) -> None:
    tool: JSON = {"type": "function", "function": {"name": "wanted"}}
    named: JSON = {"type": "function", "function": {"name": "wanted"}}
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": [tool],
        "tool_choice": named,
        "temperature": 0,
        "logit_bias": {"2": 100, "3": -100, "4": 80},
        "max_completion_tokens": 4,
    }
    config = ContinuousBatchingConfig(slot_count=1, max_context_length=32, prefill_chunk_size=3)
    with TestClient(create_app(tool_choice_model, "test", config)) as http:
        for invalid in (
            {"tools": None, "tool_choice": "required"},
            {"tools": [], "tool_choice": named},
            {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
            {"tools": [tool, tool]},
            {"tool_choice": {"type": "function", "function": {"name": "wanted", "arguments": "{}"}}},
            {"tool_choice": {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": []}}},
            {
                "tool_choice": {
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": "auto",
                        "tools": [{"type": "function", "function": {"name": "missing"}}],
                    },
                }
            },
            {
                "tool_choice": {
                    "type": "allowed_tools",
                    "allowed_tools": {"mode": "auto", "tools": [{"type": "function"}]},
                }
            },
            {
                "tool_choice": {
                    "type": "allowed_tools",
                    "allowed_tools": {"mode": "auto", "tools": [{"type": "custom", "custom": {"name": "wanted"}}]},
                }
            },
            {"tool_choice": {"type": "allowed_tools", "allowed_tools": {"mode": "none", "tools": []}}},
        ):
            response = http.post("/v1/chat/completions", json={**body, **invalid})
            assert response.status_code == 400
            assert response.json()["error"]["type"] == "invalid_request_error"
        assert http.get("/health").status_code == 200
        response = http.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "wanted"


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["auto", "required"])
def test_allowed_subset_constrains_native_calls_and_each_http_choice(
    tool_choice_model: LanguageModel, stream: bool, mode: str
) -> None:
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": [{"type": "function", "function": {"name": name}} for name in ("wanted", "other")],
        "tool_choice": {
            "type": "allowed_tools",
            "allowed_tools": {"mode": mode, "tools": [{"type": "function", "function": {"name": "wanted"}}]},
        },
        "temperature": 0,
        "logit_bias": {"2": 100 if mode == "required" else 80, "3": -100, "4": 90, "5": 95, "6": 95},
        "parallel_tool_calls": False,
        "n": 2,
        "max_completion_tokens": 4,
        "stream": stream,
        "logprobs": True,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    config = ContinuousBatchingConfig(slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=3)
    with TestClient(create_app(tool_choice_model, "test", config)) as http:
        response = http.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        if stream:
            chunks = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            choices = [choice for chunk in chunks for choice in chunk["choices"]]
            usage = chunks[-1]["usage"]
            calls = []
            for index in range(2):
                row = [choice for choice in choices if choice["index"] == index]
                assert row[-1]["finish_reason"] == "tool_calls"
                assert "".join(choice["delta"].get("content", "") for choice in row) == ""
                row_calls = [call for choice in row for call in choice["delta"].get("tool_calls", [])]
                assert len(row_calls) == 1 and row_calls[0]["index"] == 0
                assert all(not choice["logprobs"] or not choice["logprobs"]["content"] for choice in row)
                calls.extend(row_calls)
        else:
            payload = response.json()
            assert {choice["index"] for choice in payload["choices"]} == {0, 1}
            assert all(choice["finish_reason"] == "tool_calls" for choice in payload["choices"])
            assert all(choice["logprobs"]["content"] == [] for choice in payload["choices"])
            assert all(len(choice["message"]["tool_calls"]) == 1 for choice in payload["choices"])
            calls = [choice["message"]["tool_calls"][0] for choice in payload["choices"]]
            usage = payload["usage"]
        assert len({call["id"] for call in calls}) == 2
        assert all(call["function"] == {"name": "wanted", "arguments": "{}"} for call in calls)
        assert usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        if mode == "required":
            cutoff = http.post(
                "/v1/chat/completions",
                json={
                    **body,
                    "stream": False,
                    "stream_options": None,
                    "n": 1,
                    "max_completion_tokens": 1,
                    "logit_bias": {"2": 100, "7": 95, "4": 80, "5": 99},
                },
            )
            assert cutoff.status_code == 200
            (choice,) = cutoff.json()["choices"]
            assert choice["finish_reason"] == "length"
            assert not choice["message"].get("tool_calls")
            assert cutoff.json()["usage"]["completion_tokens"] == 1


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
def test_allowed_auto_can_stop_or_return_native_text_with_raw_scores(
    tool_choice_model: LanguageModel, stream: bool
) -> None:
    model = tool_choice_model
    token_id = 3
    native = "plain answer"
    if model.token_codec.config.tool_call_format is ToolCallFormat.MUSE_ATEM:
        native = " to=user<|message|>plain answer<|eot|>"
        model.token_codec.tokenizer.add_tokens([native])
        token_id = 11
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": [{"type": "function", "function": {"name": name}} for name in ("wanted", "other")],
        "tool_choice": {
            "type": "allowed_tools",
            "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "wanted"}}]},
        },
        "temperature": 0,
        "max_completion_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    config = ContinuousBatchingConfig(slot_count=1, max_context_length=32, prefill_batch_size=1, prefill_chunk_size=3)
    with TestClient(create_app(model, "test", config)) as http:
        for text in (False, True):
            response = http.post(
                "/v1/chat/completions",
                json={
                    **body,
                    "logit_bias": {"2": 80 if text else 100, "5": 95, "6": 95, str(token_id): 100 if text else 0},
                },
            )
            assert response.status_code == 200
            if stream:
                chunks = [
                    json.loads(line.removeprefix("data: "))
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                choices = [choice for chunk in chunks for choice in chunk["choices"]]
                assert not any(choice["delta"].get("tool_calls") for choice in choices)
                content = "".join(choice["delta"].get("content", "") for choice in choices)
                entries = [
                    entry for choice in choices if choice["logprobs"] for entry in choice["logprobs"]["content"]
                ]
                finish = choices[-1]["finish_reason"]
                usage = chunks[-1]["usage"]
            else:
                payload = response.json()
                (choice,) = payload["choices"]
                assert not choice["message"].get("tool_calls")
                content, entries = choice["message"]["content"], choice["logprobs"]["content"]
                finish, usage = choice["finish_reason"], payload["usage"]
            assert finish == ("length" if text else "stop")
            assert content == ("plain answer" if text else "")
            assert usage == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            if text:
                (entry,) = entries
                row = dense_log_softmax_rows(model, (1,), [token_id])[0]
                expected = float(row[token_id]) if token_id in row.argsort()[-20:] else -9999.0
                assert entry["bytes"] == list(native.encode())
                assert entry["logprob"] == pytest.approx(expected, rel=1e-4, abs=1e-3)
                np.testing.assert_allclose(
                    [alternative["logprob"] for alternative in entry["top_logprobs"]],
                    sorted(row, reverse=True)[:20],
                    rtol=1e-4,
                    atol=1e-3,
                )
            else:
                assert entries == []


@pytest.mark.fast
def test_allowed_subsets_preserve_full_prompt_during_reuse_and_empty_selection(
    tool_choice_model: LanguageModel,
) -> None:
    tokenizer = Tokenizer.from_str(tool_choice_model.token_codec.tokenizer.to_str())
    tokenizer.pre_tokenizer = Whitespace()
    codec = replace(
        tool_choice_model.token_codec.config,
        prompt_template="{% for tool in tools %}{{ tool.function.name }} {% endfor %}" + "prompt " * 11,
    ).init(tokenizer)
    model = LanguageModel(
        config=replace(tool_choice_model.config, token_codec_config=codec.config),
        token_codec=codec,
        decoder=tool_choice_model.decoder,
        sharding_config=tool_choice_model.sharding_config,
    )
    tools: list[JSON] = [{"type": "function", "function": {"name": name}} for name in ("wanted", "other")]
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": tools,
        "temperature": 0,
        "max_completion_tokens": 1,
    }
    config = ContinuousBatchingConfig(slot_count=1, max_context_length=32, prefill_batch_size=1, prefill_chunk_size=3)
    with TestClient(create_app(model, "test", config)) as http:
        for names in (["wanted"], ["other"], [], ["wanted", "wanted"]):
            choice: JSON = {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required" if names else "auto",
                    "tools": [{"type": "function", "function": {"name": name}} for name in names],
                },
            }
            response = http.post(
                "/v1/chat/completions",
                json={**body, "tool_choice": choice, "logit_bias": {"2": -100, "4": 80, "5": 90}},
            )
            assert response.status_code == 200
            payload = response.json()
            (result,) = payload["choices"]
            assert payload["usage"] == {"prompt_tokens": 13, "completion_tokens": 1, "total_tokens": 14}
            assert result["finish_reason"] == "length"
            if names:
                assert len(result["message"]["tool_calls"]) == 1
                assert result["message"]["tool_calls"][0]["function"]["name"] == names[0]
            else:
                assert not result["message"].get("tool_calls")
                assert result["message"]["content"] == codec.decode_tokens([5])
        assert http.get("/health").status_code == 200


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
def test_refusal_history_matches_plain_assistant_text_in_real_generation(
    recurrent_model: LanguageModel, stream: bool
) -> None:
    vocabulary = {"[UNK]": 0, "prompt": 1, "before": 2, "refusal": 3, "after": 4, "拒否": 5, "answer": 6}
    vocabulary.update({f"word{index}": index for index in range(7, recurrent_model.decoder.vocab_size)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = Fuse()
    codec = replace(recurrent_model.token_codec.config, prompt_template="{{ messages[1].content }} prompt").init(
        tokenizer
    )
    model = LanguageModel(
        config=replace(recurrent_model.config, token_codec_config=codec.config),
        token_codec=codec,
        decoder=recurrent_model.decoder,
        sharding_config=recurrent_model.sharding_config,
    )
    config = ContinuousBatchingConfig(slot_count=1, max_context_length=32, prefill_batch_size=1, prefill_chunk_size=3)
    body: dict[str, JSON] = {
        "model": "test",
        "temperature": 0,
        "max_completion_tokens": 2,
        "logit_bias": {"6": 100},
        "logprobs": True,
        "top_logprobs": 20,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True, "include_obfuscation": False}
    cases: tuple[tuple[dict[str, JSON], str], ...] = (
        ({"content": None, "refusal": "before refusal after"}, "before refusal after"),
        ({"refusal": "拒否"}, "拒否"),
        ({"content": "before ", "refusal": "refusal after"}, "before refusal after"),
        ({"content": [{"type": "refusal", "refusal": "before refusal after"}]}, "before refusal after"),
        (
            {
                "content": [{"type": "text", "text": "before "}, {"type": "text", "text": "refusal "}],
                "refusal": "after",
            },
            "before refusal after",
        ),
        ({"content": "before refusal after", "refusal": None}, "before refusal after"),
        ({"content": None, "refusal": ""}, ""),
        ({"content": [{"type": "refusal", "refusal": ""}]}, ""),
    )
    with TestClient(create_app(model, "test", config)) as http:
        for history, text in cases:
            payloads = []
            for assistant in ({"content": text}, history, {**history, "audio": None}):
                response = http.post(
                    "/v1/chat/completions",
                    json={
                        **body,
                        "messages": [
                            {"role": "user", "content": "prompt"},
                            {"role": "assistant", **assistant},
                            {"role": "user", "content": "prompt"},
                        ],
                    },
                )
                assert response.status_code == 200
                if stream:
                    payloads.append(
                        [
                            json.loads(line.removeprefix("data: "))["choices"]
                            for line in response.text.splitlines()
                            if line.startswith("data: ") and line != "data: [DONE]"
                        ]
                    )
                else:
                    payloads.append(response.json()["choices"])
            assert payloads[0] == payloads[1] == payloads[2]

        invalid: list[dict[str, JSON]] = [
            {"role": "assistant", "content": None, "refusal": None},
            {"role": "assistant", "content": []},
            {"role": "assistant", "content": "before", "refusal": 7},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": None}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "before"}, {"type": "refusal", "refusal": "after"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "before"}, {"type": "refusal", "refusal": "after"}],
            },
        ]
        for role in ("user", "system", "developer", "tool"):
            invalid.extend(
                (
                    {"role": role, "content": "before", "refusal": ""},
                    {"role": role, "content": [{"type": "refusal", "refusal": "after"}]},
                )
            )
        for message in invalid:
            response = http.post("/v1/chat/completions", json={**body, "messages": [message]})
            assert response.status_code == 400
            assert response.json()["error"]["type"] == "invalid_request_error"
        assert (
            http.post(
                "/v1/chat/completions",
                json={
                    **body,
                    "messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": ""}],
                },
            ).status_code
            == 200
        )


@pytest.mark.fast
def test_sdk_refusal_message_replays_through_the_real_chat_api(recurrent_model: LanguageModel) -> None:
    codec = replace(recurrent_model.token_codec.config, prompt_template="{{ messages[1].content }} prompt").init(
        recurrent_model.token_codec.tokenizer
    )
    model = LanguageModel(
        config=replace(recurrent_model.config, token_codec_config=codec.config),
        token_codec=codec,
        decoder=recurrent_model.decoder,
        sharding_config=recurrent_model.sharding_config,
    )
    config = ContinuousBatchingConfig(slot_count=1, max_context_length=32, prefill_batch_size=1, prefill_chunk_size=3)

    async def replay(http: TestClient) -> None:
        async with AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=http.app, raise_app_exceptions=False), trust_env=False
            ),
        ) as client:
            message = ChatCompletionMessage(role="assistant", content=None, refusal="I cannot help.")
            completion = await client.chat.completions.create(
                model="test",
                messages=[
                    {"role": "user", "content": "prompt"},
                    cast("ChatCompletionAssistantMessageParam", message),
                    {"role": "user", "content": "prompt"},
                ],
                temperature=0,
                max_completion_tokens=1,
            )
            assert completion.choices[0].finish_reason == "length"
            assert completion.choices[0].message.refusal is None

    with TestClient(create_app(model, "test", config)) as http:
        asyncio.run(replay(http))


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
def test_strict_native_tools_preserve_unicode_schema_values_and_references(
    tool_choice_model: LanguageModel, stream: bool
) -> None:
    raw: dict[str, JSON] = {"$ref": "#/生データ", "$defs": {"型": "雪😀"}, "properties": {"値": "é"}}
    definitions: dict[str, JSON] = {
        "型": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {"内側": {"$ref": "#/$defs/型"}},
                    "required": ["内側"],
                    "additionalProperties": False,
                },
            ]
        }
    }
    finite: dict[str, JSON] = {"title": "雪😀", "default": [2**63 - 1, True, None], "$ref": "#/生データ"}
    finite_object: dict[str, JSON] = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "default": {
                "type": "array",
                "items": {"anyOf": [{"type": "integer"}, {"type": "boolean"}, {"type": "null"}]},
            },
            "$ref": {"type": "string"},
        },
        "required": ["title", "default", "$ref"],
        "additionalProperties": False,
        "default": raw,
        "examples": [raw],
    }
    finite_array: dict[str, JSON] = {
        "type": "array",
        "items": {"anyOf": [{"type": "integer"}, {"type": "boolean"}, {"type": "null"}]},
    }
    cases: tuple[tuple[str, dict[str, JSON], JSON], ...] = (
        ("値", {"type": "integer", "const": 1}, 1),
        ("値", {"type": "string", "const": "雪😀", "default": raw, "examples": [raw]}, "雪😀"),
        ("値", {"$ref": "#/$defs/型"}, {"内側": {"内側": None}}),
        ("値", {"anyOf": [{"type": "null"}, {"$ref": "#"}]}, {"値": {"値": None}}),
        ("値", {"anyOf": [{"$ref": "#/$defs/型"}, {"$ref": "#"}]}, {"値": {"値": None}}),
        ("値", {**finite_object, "const": finite}, finite),
        ("値", {**finite_object, "enum": [finite, {**finite, "title": "別"}]}, finite),
        ("値", {**finite_array, "const": finite["default"]}, finite["default"]),
        ("値", {**finite_array, "enum": [finite["default"], [None, True]]}, finite["default"]),
        ("x/y", {"type": "integer", "const": 1}, 1),
        ("x~y", {"type": "integer", "const": 1}, 1),
    )
    tool_format = tool_choice_model.token_codec.config.tool_call_format
    empty_call = tool_choice_model.token_codec.decode_tokens([4])
    for key, node, expected in cases:
        if tool_format is ToolCallFormat.LIQUID and key in ("x/y", "x~y"):
            continue
        value = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        wrong = json.dumps("wrong")
        if tool_format is ToolCallFormat.QWEN_XML:
            valid = empty_call.replace("</function>", f"<parameter={key}>{value}</parameter></function>")
            invalid = empty_call.replace("</function>", f"<parameter={key}>{wrong}</parameter></function>")
        elif tool_format is ToolCallFormat.LIQUID:
            valid = empty_call.replace("wanted()", f"wanted({key}={value})")
            invalid = empty_call.replace("wanted()", f"wanted({key}={wrong})")
        else:
            valid = empty_call.replace(
                "</atem:invoke>", f'<atem:parameter name="{key}">{value}</atem:parameter></atem:invoke>'
            )
            invalid = empty_call.replace(
                "</atem:invoke>", f'<atem:parameter name="{key}">{wrong}</atem:parameter></atem:invoke>'
            )
        tokenizer = Tokenizer(
            WordLevel(
                {
                    "[UNK]": 0,
                    "prompt": 1,
                    "<eos>": 2,
                    "plain": 3,
                    valid: 4,
                    invalid: 5,
                    **{f"unused{index}": index for index in range(6, tool_choice_model.decoder.vocab_size)},
                },
                unk_token="[UNK]",
            )
        )
        tokenizer.decoder = Fuse()
        codec = tool_choice_model.token_codec.config.init(tokenizer)
        model = replace(
            tool_choice_model,
            token_codec=codec,
            config=replace(tool_choice_model.config, token_codec_config=codec.config),
        )
        parameters: dict[str, JSON] = {
            "type": "object",
            "properties": {key: node},
            "required": [key],
            "additionalProperties": False,
            "$defs": definitions,
        }
        body: dict[str, JSON] = {
            "model": "test",
            "messages": [{"role": "user", "content": "prompt"}],
            "tools": [{"type": "function", "function": {"name": "wanted", "strict": True, "parameters": parameters}}],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": 0,
            "logit_bias": {"4": 90, "5": 100, "2": -100},
            "max_completion_tokens": 1,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True, "include_obfuscation": False}
        config = ContinuousBatchingConfig(
            slot_count=1, max_context_length=16, prefill_batch_size=1, prefill_chunk_size=3
        )
        with TestClient(create_app(model, "test", config)) as http:
            response = http.post("/v1/chat/completions", json=body)
            assert response.status_code == 200, response.text
            if stream:
                chunks = [
                    json.loads(line.removeprefix("data: "))
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                calls = [
                    call
                    for chunk in chunks
                    for choice in chunk["choices"]
                    for call in choice["delta"].get("tool_calls", ())
                ]
                arguments = "".join(call["function"].get("arguments", "") for call in calls)
                assert calls[0]["function"]["name"] == "wanted"
            else:
                (call,) = response.json()["choices"][0]["message"]["tool_calls"]
                assert call["function"]["name"] == "wanted"
                arguments = call["function"]["arguments"]
            assert json.loads(arguments) == {key: expected}
            assert http.get("/health").status_code == 200


@pytest.mark.fast
@pytest.mark.parametrize("stream", [False, True])
def test_json_responses_preserve_unicode_definitions_and_recursive_root(
    recurrent_model: LanguageModel, stream: bool
) -> None:
    recursive: dict[str, JSON] = {
        "type": "object",
        "properties": {"値": {"anyOf": [{"type": "null"}, {"$ref": "#"}]}},
        "required": ["値"],
        "additionalProperties": False,
    }
    references: dict[str, JSON] = {
        "type": "object",
        "properties": {"値": {"$ref": "#/$defs/型"}},
        "required": ["値"],
        "additionalProperties": False,
        "$defs": {
            "型": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {"内側": {"$ref": "#/$defs/型"}},
                        "required": ["内側"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
    }
    annotation: dict[str, JSON] = {"$ref": "#/生データ", "$defs": {"型": "雪😀"}, "properties": {"値": "é"}}
    finite: dict[str, JSON] = {"title": "雪😀", "default": [2**63 - 1, True, None], "$ref": "#/生データ"}
    finite_object: dict[str, JSON] = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "default": {
                "type": "array",
                "items": {"anyOf": [{"type": "integer"}, {"type": "boolean"}, {"type": "null"}]},
            },
            "$ref": {"type": "string"},
        },
        "required": ["title", "default", "$ref"],
        "additionalProperties": False,
        "default": annotation,
        "examples": [annotation],
    }
    finite_array: dict[str, JSON] = {
        "type": "array",
        "items": {"anyOf": [{"type": "integer"}, {"type": "boolean"}, {"type": "null"}]},
    }
    cases: tuple[tuple[dict[str, JSON], dict[str, JSON]], ...] = (
        (recursive, {"値": {"値": None}}),
        (references, {"値": {"内側": None}}),
        ({**references, "properties": {"値": {**finite_object, "const": finite}}}, {"値": finite}),
        (
            {**references, "properties": {"値": {**finite_object, "enum": [finite, {**finite, "title": "別"}]}}},
            {"値": finite},
        ),
        (
            {**references, "properties": {"値": {**finite_array, "const": finite["default"]}}},
            {"値": finite["default"]},
        ),
        (
            {**references, "properties": {"値": {**finite_array, "enum": [finite["default"], [None, True]]}}},
            {"値": finite["default"]},
        ),
    )
    for schema, expected in cases:
        raw = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        tokenizer = Tokenizer(
            WordLevel(
                {
                    "[UNK]": 0,
                    "prompt": 1,
                    raw: 4,
                    **{
                        f"unused{index}": index for index in range(2, recurrent_model.decoder.vocab_size) if index != 4
                    },
                },
                unk_token="[UNK]",
            )
        )
        tokenizer.decoder = Fuse()
        codec = replace(recurrent_model.token_codec.config, prompt_template="prompt").init(tokenizer)
        model = replace(
            recurrent_model, token_codec=codec, config=replace(recurrent_model.config, token_codec_config=codec.config)
        )
        body: dict[str, JSON] = {
            "model": "test",
            "messages": [{"role": "user", "content": "Reply in JSON."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Result", "strict": True, "schema": schema},
            },
            "temperature": 0,
            "logit_bias": {"4": 100},
            "max_completion_tokens": 1,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True, "include_obfuscation": False}
        config = ContinuousBatchingConfig(
            slot_count=1, max_context_length=16, prefill_batch_size=1, prefill_chunk_size=3
        )
        with TestClient(create_app(model, "test", config)) as http:
            response = http.post("/v1/chat/completions", json=body)
            assert response.status_code == 200, response.text
            if stream:
                chunks = [
                    json.loads(line.removeprefix("data: "))
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                content = "".join(
                    choice["delta"].get("content", "") for chunk in chunks for choice in chunk["choices"]
                )
            else:
                content = response.json()["choices"][0]["message"]["content"]
            assert content == raw
            assert json.loads(content) == expected
            assert http.get("/health").status_code == 200
