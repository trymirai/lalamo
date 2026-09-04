import asyncio
import random
from dataclasses import dataclass
from typing import Any, cast

import httpx2
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from openai import AsyncOpenAI, AsyncStream, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionUserMessageParam

from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    GeneratedToken,
    SequenceFinished,
    TokenEvent,
)
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ReasoningEffort, UserMessage
from lalamo.module import ForwardPassMode, Keychain, ShardingConfig
from lalamo.modules import DecoderForwardPassConfig
from lalamo.server import create_app
from tests.conftest import ConvertModel

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_standard_openai_client_chat_completions_streaming_errors_and_concurrency(
    _convert_model_session: ConvertModel,
) -> None:
    model = LanguageModel.load(
        _convert_model_session("Qwen/Qwen3.5-0.8B", cached=True), sharding_config=ShardingConfig.replicated()
    )
    assert isinstance(model, LanguageModel)

    unicode_token_ids = model.token_codec.encode_text("👩‍💻")
    unicode_decoder = model.token_codec.decode_stream(ReasoningEffort.NO_REASONING)
    assert "".join(unicode_decoder.step(token_id) for token_id in unicode_token_ids) == "👩‍💻"
    protocol_token_ids = model.token_codec.encode_text("private\n</think>\n\npublic")
    protocol_decoder = model.token_codec.decode_stream(ReasoningEffort.MEDIUM)
    assert "".join(protocol_decoder.step(token_id) for token_id in protocol_token_ids) == "public"
    assert protocol_decoder.finish() == "public"

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

    async def run() -> None:
        api = create_app(
            model, "org/test-model", ContinuousBatchingConfig(total_pages=16, slot_count=4, prefill_batch_size=4)
        )
        async with (
            api.router.lifespan_context(api),
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=api)) as http,
        ):
            client = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=http)
            assert (await client.models.retrieve("org/test-model")).id == "org/test-model"

            async def complete(**overrides: Any) -> ChatCompletion:  # noqa: ANN401
                return cast("ChatCompletion", await client.chat.completions.create(**request, **overrides))

            chat, concurrent = await asyncio.gather(
                complete(logprobs=True, top_logprobs=20, metadata={"test": "api"}), complete()
            )
            assert concurrent.choices[0].finish_reason == "length"
            assert concurrent.choices[0].message.content == chat.choices[0].message.content
            chat_logprobs = chat.choices[0].logprobs
            assert chat_logprobs is not None and chat_logprobs.content is not None
            token = chat_logprobs.content[0]
            unique = {item.token: item.logprob for item in token.top_logprobs} | {token.token: token.logprob}
            assert 0 <= 1 - np.exp(tuple(unique.values())).sum() <= 1
            stop = chat.choices[0].message.content
            assert stop and len(chat_logprobs.content) > 1

            stream = await client.chat.completions.create(
                **request,
                logprobs=True,
                top_logprobs=20,
                stream=True,
                stream_options={"include_usage": True, "include_obfuscation": False},
            )
            chunks = [cast("ChatCompletionChunk", chunk) async for chunk in cast("AsyncStream", stream)]
            streamed = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
            streamed_logprobs = [
                entry
                for chunk in chunks
                if chunk.choices and chunk.choices[0].logprobs
                for entry in chunk.choices[0].logprobs.content or []
            ]
            assert streamed == stop and len(streamed_logprobs) == len(chat_logprobs.content)
            assert chunks[-2].choices[0].finish_reason == "length"
            assert all(chunk.usage is None for chunk in chunks[:-1]) and chunks[-1].usage is not None

            stopped = await complete(stop=stop)
            assert stopped.choices[0].message.content == "" and stopped.choices[0].finish_reason == "stop"
            with pytest.raises(BadRequestError):
                await client.chat.completions.create(
                    model="org/test-model", messages=[{"role": "user", "content": "x"}], top_logprobs=1
                )

    asyncio.run(run())


@dataclass(frozen=True)
class FuzzRequest:
    prompt: tuple[int, ...]
    max_output_length: int
    stop_token_ids: tuple[int, ...]
    arrival_step: int
    return_logprobs: bool


def _dense_log_softmax_rows(model: LanguageModel, prompt: tuple[int, ...], token_ids: list[int]) -> list[np.ndarray]:
    """Teacher-forced dense reference: the log-softmax row before every token and one more after the last."""
    keychain = Keychain.init(0, sharding_config=model.sharding_config)
    prefilled = model.prefill_tokens(jnp.asarray([prompt]), len(prompt) + len(token_ids), keychain=keychain)
    state, logits = prefilled.state, prefilled.last_token_logits
    rows = []
    for position, token_id in enumerate(token_ids, start=len(prompt)):
        rows.append(np.asarray(jax.nn.log_softmax(logits[0].astype(jnp.float32))))
        decoded = model.decoder(
            jnp.asarray([[token_id]]),
            jnp.asarray([[position]]),
            state=state,
            return_updated_state=True,
            forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
            keychain=keychain,
        )
        assert decoded.updated_state is not None
        state, logits = decoded.updated_state, decoded.logits[:, 0]
    return [*rows, np.asarray(jax.nn.log_softmax(logits[0].astype(jnp.float32)))]


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
        total_pages=rng.randint(3, 12),
        slot_count=rng.randint(1, 4),
        prefill_batch_size=rng.randint(1, 3),
        prefill_chunk_size=rng.choice([32, 64, 128]),
    )
    engine = ContinuousBatchingEngine(model, config)
    source = model.token_codec.encode_request([UserMessage("one two three four five six seven eight " * 64)])
    keychain = Keychain.init(0, sharding_config=model.sharding_config)

    requests = []
    for _ in range(rng.randint(1, 8)):
        max_output_length = rng.randint(1, 40)
        prompt = tuple(source[: rng.randint(1, engine.context_limit - max_output_length)])
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
    assert len(engine._free_pages) == config.total_pages and len(engine._free_slots) == config.slot_count  # noqa: SLF001

    for request, sequence_events in zip(requests, events, strict=True):
        *tokens, finished = sequence_events
        assert isinstance(finished, SequenceFinished)
        token_ids = [event.token_id for event in tokens if isinstance(event, GeneratedToken)]
        assert len(token_ids) == len(tokens) and not set(token_ids) & set(request.stop_token_ids)
        if finished.reason is FinishReason.STOP:
            assert len(token_ids) < request.max_output_length and finished.completion_tokens == len(token_ids) + 1
        else:
            assert len(token_ids) == request.max_output_length == finished.completion_tokens

        rows = _dense_log_softmax_rows(model, request.prompt, token_ids)
        (stop_token_id,) = request.stop_token_ids or (int(rows[-1].argmax()),)
        chosen = [*token_ids, stop_token_id] if finished.reason is FinishReason.STOP else token_ids
        greedy_gaps = [float(row.max() - row[token_id]) for row, token_id in zip(rows, chosen, strict=False)]
        assert max(greedy_gaps) < 0.25, (model_name, seed, greedy_gaps)
        for row, event in zip(rows, tokens, strict=False):
            assert isinstance(event, GeneratedToken)
            if event.logprobs is None:
                assert not request.return_logprobs
                continue
            assert event.token_id == event.logprobs.top_token_ids[0]
            assert abs(event.logprobs.logprob - row[event.token_id]) < 1.0
            assert (
                np.max(np.abs(np.asarray(event.logprobs.top_logprobs) - row[list(event.logprobs.top_token_ids)])) < 1.0
            )
