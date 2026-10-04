import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, cast

import httpx2
import jax.numpy as jnp
import numpy as np
import pytest
from openai import AsyncOpenAI, AsyncStream, BadRequestError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    GeneratedToken,
    SequenceFinished,
    TokenEvent,
)
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import AssistantMessage, ReasoningEffort, UserMessage
from lalamo.module import Keychain, ShardingConfig
from lalamo.server import create_app
from tests.conftest import ConvertModel
from tests.helpers import dense_log_softmax_rows

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_standard_openai_client_chat_completions_streaming_tools_and_concurrency(
    _convert_model_session: ConvertModel,
) -> None:
    model = LanguageModel.load(
        _convert_model_session("Qwen/Qwen3.5-0.8B", cached=True), sharding_config=ShardingConfig.replicated()
    )
    assert isinstance(model, LanguageModel)
    codec = model.token_codec

    unicode_decoder = codec.decode_stream(
        codec.render_request([UserMessage("")], reasoning_effort=ReasoningEffort.NO_REASONING)
    )
    assert "".join(unicode_decoder.step(token_id)[1] for token_id in codec.encode_text("👩‍💻")) == "👩‍💻"
    protocol_decoder = codec.decode_stream(
        codec.render_request([UserMessage("")], reasoning_effort=ReasoningEffort.MEDIUM)
    )
    pieces = [protocol_decoder.step(token_id) for token_id in codec.encode_text("private\n</think>\n\npublic")]
    assert "".join(reasoning for reasoning, _ in pieces) == "private\n"
    assert "".join(response for _, response in pieces) == "public"
    assert protocol_decoder.finish()[2] == AssistantMessage(chain_of_thought="private\n", response="public")

    request: dict[str, Any] = {
        "model": "org/test-model",
        "messages": [ChatCompletionUserMessageParam(role="user", content=[{"type": "text", "text": "Say hi."}])],
        "max_completion_tokens": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "seed": 7,
        "reasoning_effort": "none",
        "extra_body": {"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1},
    }
    weather_tool: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}},
                "required": ["city"],
            },
        },
    }

    async def run() -> None:
        api = create_app(
            model, "org/test-model", ContinuousBatchingConfig(total_pages=16, slot_count=4, prefill_batch_size=4)
        )
        async with (
            api.router.lifespan_context(api),
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=api)) as http,
        ):
            client = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=http)

            async def complete(**overrides: Any) -> ChatCompletion:  # noqa: ANN401
                return cast("ChatCompletion", await client.chat.completions.create(**(request | overrides)))

            chat, concurrent = await asyncio.gather(complete(logprobs=True, top_logprobs=20), complete())
            assert concurrent.choices[0].finish_reason == "length"
            assert concurrent.choices[0].message.content == chat.choices[0].message.content
            chat_logprobs = chat.choices[0].logprobs
            assert chat_logprobs is not None and chat_logprobs.content is not None
            token = chat_logprobs.content[0]
            unique = {item.token: item.logprob for item in token.top_logprobs} | {token.token: token.logprob}
            assert 0 <= 1 - np.exp(tuple(unique.values())).sum() <= 1
            stop = chat.choices[0].message.content
            assert stop and len(chat_logprobs.content) == 3

            stopped = await complete(stop=stop)
            assert stopped.choices[0].message.content == "" and stopped.choices[0].finish_reason == "stop"

            thinking = await complete(
                reasoning_effort=None, extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
            thinking_message = thinking.choices[0].message
            assert thinking_message.content == "" and thinking_message.model_extra
            assert len(thinking_message.model_extra["reasoning_content"]) > 0
            thinking_stream = await client.chat.completions.create(
                **request
                | {"reasoning_effort": None, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
                stream=True,
            )
            thinking_chunks = [
                cast("ChatCompletionChunk", chunk) async for chunk in cast("AsyncStream", thinking_stream)
            ]
            streamed_reasoning = "".join(
                chunk.choices[0].delta.model_extra.get("reasoning_content") or ""
                for chunk in thinking_chunks
                if chunk.choices and chunk.choices[0].delta.model_extra
            )
            assert streamed_reasoning == thinking_message.model_extra["reasoning_content"]

            question = ChatCompletionUserMessageParam(
                role="user", content="What's the weather in Paris for the next 3 days?"
            )
            tool_request: dict[str, Any] = {
                "model": "org/test-model",
                "tools": [weather_tool],
                "temperature": 0,
                "max_completion_tokens": 256,
                "reasoning_effort": "none",
            }
            called = cast("ChatCompletion", await client.chat.completions.create(messages=[question], **tool_request))
            message = called.choices[0].message
            assert called.choices[0].finish_reason == "tool_calls" and message.tool_calls
            (call,) = message.tool_calls
            assert isinstance(call, ChatCompletionMessageFunctionToolCall)
            assert call.function.name == "get_weather"
            assert json.loads(call.function.arguments) == {"city": "Paris", "days": 3}
            answered = cast(
                "ChatCompletion",
                await client.chat.completions.create(
                    messages=[
                        question,
                        cast("Any", message.model_dump(exclude_none=True)),
                        {"role": "tool", "tool_call_id": call.id, "content": '{"forecast": "sunny"}'},
                    ],
                    **tool_request,
                ),
            )
            assert answered.choices[0].message.content

            with pytest.raises(BadRequestError):
                await complete(response_format={"type": "json_object"})

    asyncio.run(run())


@pytest.mark.parametrize("cancel_active", [False, True], ids=["before_admission", "after_prefill"])
def test_engine_cancellation_allows_new_request(_convert_model_session: ConvertModel, cancel_active: bool) -> None:
    paged_language_model = LanguageModel.load(
        _convert_model_session("Qwen/Qwen3.5-0.8B", cached=True), sharding_config=ShardingConfig.replicated()
    )
    engine = ContinuousBatchingEngine(
        paged_language_model, ContinuousBatchingConfig(total_pages=8, slot_count=1, max_context_length=256)
    )
    generation_config = GenerationConfig(temperature=0.0)
    canceled_events: list[TokenEvent] = []
    canceled = engine.submit(
        tuple(paged_language_model.token_codec.encode_request([UserMessage("Say hi.")])),
        64,
        generation_config,
        0,
        on_events=canceled_events.extend,
    )
    if cancel_active:
        assert engine.step()
    canceled.set()

    prompt = tuple(paged_language_model.token_codec.encode_request([UserMessage("Name a fruit.")]))
    events: list[TokenEvent] = []
    engine.submit(prompt, 4, generation_config, 0, on_events=events.extend)
    for _ in range(8):
        if not engine.step():
            break
    assert not engine.step()
    assert canceled_events == []
    assert events[-1] == SequenceFinished(FinishReason.LENGTH, 4)
    expected = paged_language_model.stream_tokens(
        jnp.asarray(prompt),
        generation_config,
        4,
        keychain=Keychain.init(0, sharding_config=paged_language_model.sharding_config),
    )
    assert [event.token_id for event in events if isinstance(event, GeneratedToken)] == list(map(int, expected))


@dataclass(frozen=True)
class FuzzRequest:
    prompt: tuple[int, ...]
    max_output_length: int
    stop_token_ids: tuple[int, ...]
    arrival_step: int
    return_logprobs: bool


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3.5-0.8B", "google/gemma-3-1b-it"])
@pytest.mark.parametrize("seed", range(4))
def test_fuzz_engine_matches_dense_greedy_decoding(
    _convert_model_session: ConvertModel, model_name: str, seed: int
) -> None:
    model = LanguageModel.load(
        _convert_model_session(model_name, cached=True), sharding_config=ShardingConfig.replicated()
    )
    rng = random.Random(seed)
    config = ContinuousBatchingConfig(
        total_pages=rng.randint(6, 12),
        slot_count=rng.randint(2, 4),
        max_context_length=96,
        prefill_batch_size=rng.randint(1, 3),
        prefill_chunk_size=rng.choice([24, 32, 48, 64]),
    )
    engine = ContinuousBatchingEngine(model, config)
    source = model.token_codec.encode_request([UserMessage("one two three four five six seven eight " * 64)])
    keychain = Keychain.init(0, sharding_config=model.sharding_config)

    requests = []
    for _ in range(rng.randint(3, 8)):
        max_output_length = rng.randint(1, 40)
        prompt = tuple(source[: rng.randint(1, engine.context_limit - max_output_length)])
        if requests and rng.random() < 0.5:
            # Extend an earlier prompt so that the finished prompt's cached prefix can be reused.
            base = rng.choice(requests).prompt
            longest = engine.context_limit - max_output_length
            if len(base) < longest:
                prompt = tuple(source[: rng.randint(len(base) + 1, longest)])
        stop_token_ids: tuple[int, ...] = ()
        if rng.random() < 0.5:
            greedy = GenerationConfig(stop_token_ids=(), temperature=0.0)
            reference = list(model.stream_tokens(jnp.asarray(prompt), greedy, max_output_length, keychain=keychain))
            stop_token_ids = (int(rng.choice(reference)),)
        requests.append(FuzzRequest(prompt, max_output_length, stop_token_ids, rng.randint(0, 6), rng.random() < 0.7))

    events: list[list[TokenEvent]] = [[] for _ in requests]
    step = 0
    while True:
        for index, request in enumerate(requests):
            if request.arrival_step == step:
                engine.submit(
                    request.prompt,
                    request.max_output_length,
                    GenerationConfig(stop_token_ids=request.stop_token_ids, temperature=0.0),
                    index,
                    return_logprobs=request.return_logprobs,
                    on_events=events[index].extend,
                )
        busy = engine.step()
        step += 1
        if not busy and step > max(request.arrival_step for request in requests):
            break
    cached_pages = sum(len(prefix.pages) for prefix in engine._prefix_cache)  # noqa: SLF001
    assert len(engine._free_pages) + cached_pages == config.total_pages  # noqa: SLF001
    assert len(engine._free_slots) == engine.slot_count  # noqa: SLF001

    # A prompt extending a finished prompt must consume that prompt's cached prefix and still match dense decoding.
    base = max((prefix.token_ids for prefix in engine._prefix_cache), key=len)  # noqa: SLF001
    cached_count = sum(prefix.token_ids == base for prefix in engine._prefix_cache)  # noqa: SLF001
    follow_up = FuzzRequest((*base, source[len(base)]), 8, (), 0, return_logprobs=True)
    requests.append(follow_up)
    events.append([])
    engine.submit(
        follow_up.prompt,
        8,
        GenerationConfig(stop_token_ids=(), temperature=0.0),
        0,
        return_logprobs=True,
        on_events=events[-1].extend,
    )
    assert engine.step()
    assert sum(prefix.token_ids == base for prefix in engine._prefix_cache) < cached_count  # noqa: SLF001
    while not isinstance(events[-1][-1] if events[-1] else None, SequenceFinished):
        assert engine.step()

    for request, sequence_events in zip(requests, events, strict=True):
        *tokens, finished = sequence_events
        assert isinstance(finished, SequenceFinished)
        token_ids = [event.token_id for event in tokens if isinstance(event, GeneratedToken)]
        assert len(token_ids) == len(tokens) and not set(token_ids) & set(request.stop_token_ids)
        if finished.reason is FinishReason.STOP:
            assert len(token_ids) < request.max_output_length and finished.completion_tokens == len(token_ids) + 1
        else:
            assert len(token_ids) == request.max_output_length == finished.completion_tokens

        rows = dense_log_softmax_rows(model, request.prompt, token_ids)
        (stop_token_id,) = request.stop_token_ids or (int(rows[-1].argmax()),)
        chosen = [*token_ids, stop_token_id] if finished.reason is FinishReason.STOP else token_ids
        greedy_gaps = [float(row.max() - row[token_id]) for row, token_id in zip(rows, chosen, strict=False)]
        assert max(greedy_gaps) <= 0.25, (model_name, seed, greedy_gaps)
        for row, event in zip(rows, tokens, strict=False):
            assert isinstance(event, GeneratedToken)
            if event.logprobs is None:
                assert not request.return_logprobs
                continue
            assert event.token_id == event.logprobs.top_token_ids[0]
            # BF16 projections round differently with prefill shape; strict top-logprob parity uses tiny fixtures.
            assert abs(event.logprobs.logprob - row[event.token_id]) < 1.0
