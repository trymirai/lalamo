# fmt: off
import asyncio

import httpx2
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletionUserMessageParam

from lalamo.inference.continuous_batching import ContinuousBatchingConfig, ContinuousBatchingEngine, TokenEvent
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.module import ForwardPassMode, Keychain, ShardingConfig
from lalamo.modules import DecoderForwardPassConfig
from lalamo.server import create_app
from tests.conftest import ConvertModel

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
@pytest.fixture(scope="module")
def model(_convert_model_session: ConvertModel) -> LanguageModel:
    loaded = LanguageModel.load(_convert_model_session("Qwen/Qwen3.5-0.8B", cached=True),
                                sharding_config=ShardingConfig.replicated())
    assert isinstance(loaded, LanguageModel)
    return loaded
def test_standard_openai_client_chat_completions_streaming_errors_and_concurrency(model: LanguageModel) -> None:
    async def run() -> None:
        api = create_app(model, "org/test-model",
                         ContinuousBatchingConfig(page_size=32, total_pages=16, slot_count=4, prefill_batch_size=4))
        async with api.router.lifespan_context(api), \
                   httpx2.AsyncClient(transport=httpx2.ASGITransport(app=api)) as http:
            client = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=http)
            retrieved = await client.models.retrieve("org/test-model")
            assert retrieved.id == "org/test-model"
            messages = [ChatCompletionUserMessageParam(
                role="user", content=[{"type": "text", "text": "Say hi."}])]
            chat, concurrent = await asyncio.gather(
                client.chat.completions.create(
                    model="org/test-model", messages=messages, max_completion_tokens=3, temperature=0.7,
                    top_p=0.9, presence_penalty=0.1, frequency_penalty=0.1, seed=7,
                    logprobs=True, top_logprobs=20, metadata={"test": "api"}, user="test", store=False,
                    extra_body={"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1}),
                client.chat.completions.create(
                    model="org/test-model", messages=messages, max_completion_tokens=3, temperature=0.7,
                    top_p=0.9, presence_penalty=0.1, frequency_penalty=0.1, seed=7,
                    extra_body={"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1}),
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
                model="org/test-model", messages=messages, max_completion_tokens=3, temperature=0.7,
                top_p=0.9, presence_penalty=0.1, frequency_penalty=0.1, seed=7,
                logprobs=True, top_logprobs=20, stream=True,
                stream_options={"include_usage": True, "include_obfuscation": False},
                extra_body={"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1},
            )
            chunks = [chunk async for chunk in stream]
            streamed = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
            streamed_logprobs = [entry for chunk in chunks if chunk.choices and chunk.choices[0].logprobs
                                 for entry in chunk.choices[0].logprobs.content or []]
            assert streamed == stop and len(streamed_logprobs) == len(chat_logprobs.content)
            assert chunks[-2].choices[0].finish_reason == "length"
            assert all(chunk.usage is None for chunk in chunks[:-1]) and chunks[-1].usage is not None
            stopped = await client.chat.completions.create(
                model="org/test-model", messages=messages, max_completion_tokens=3, temperature=0.7,
                top_p=0.9, presence_penalty=0.1, frequency_penalty=0.1, seed=7, stop=stop,
                extra_body={"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1})
            assert stopped.choices[0].message.content == "" and stopped.choices[0].finish_reason == "stop"
            with pytest.raises(BadRequestError):
                await client.chat.completions.create(
                    model="org/test-model", messages=[{"role": "user", "content": "x"}], top_logprobs=1)
    asyncio.run(run())
def test_paged_tokens_and_normalized_top20_match_dense_across_boundaries_and_arrivals(
    _convert_model_session: ConvertModel,
) -> None:
    for model_name, prompt_lengths, output_lengths, page_size, total_pages, max_error in (
        ("Qwen/Qwen3.5-0.8B", (63, 95), (17, 17), 32, 5, 0.8),
        ("google/gemma-3-1b-it", (511, 513), (65, 33), 32, 33, 0.6),
    ):
        model = LanguageModel.load(_convert_model_session(model_name, cached=True),
                                   sharding_config=ShardingConfig.replicated())
        source = model.token_codec.encode_request([UserMessage("one two three four five six seven eight " * 256)])
        prompts = tuple(tuple(source[:length]) for length in prompt_lengths)
        generation = GenerationConfig(stop_token_ids=(), temperature=0.0)
        engine = ContinuousBatchingEngine(model,
                                          ContinuousBatchingConfig(page_size=page_size, total_pages=total_pages,
                                                                   slot_count=2, prefill_batch_size=1))
        queues = [engine.submit(prompts[0], output_lengths[0], generation, 0, return_logprobs=True)]
        assert engine.step()
        queues.append(engine.submit(prompts[1], output_lengths[1], generation, 1, return_logprobs=True))
        assert engine.step() and engine.step()
        events = [[], []]
        finished: set[int] = set()
        while len(finished) < len(queues):
            assert engine.step()
            for index, queue in enumerate(queues):
                while not queue.empty():
                    event = queue.get()
                    if event.token_id is not None:
                        events[index].append(event)
                    else:
                        finished.add(index)
        assert tuple(map(len, events)) == output_lengths
        max_logprob_error = max_greedy_gap = 0.0
        for prompt, actual in zip(prompts, events, strict=True):
            prefilled = model.prefill_tokens(jnp.asarray([prompt]), len(prompt) + len(actual),
                                             keychain=Keychain.init(0, sharding_config=model.sharding_config))
            state, logits = prefilled.state, prefilled.last_token_logits
            for position, event in enumerate(actual, start=len(prompt)):
                assert event.token_id is not None and event.logprob is not None
                normalized = np.asarray(jax.nn.log_softmax(logits[0].astype(jnp.float32)))
                assert event.token_id == event.top_token_ids[0]
                selected = normalized[np.asarray(event.top_token_ids)]
                max_logprob_error = max(max_logprob_error, abs(event.logprob - normalized[event.token_id]),
                                        float(np.max(np.abs(np.asarray(event.top_logprobs) - selected))))
                max_greedy_gap = max(max_greedy_gap, float(normalized.max() - normalized[event.token_id]))
                decoded = model.decoder(
                    jnp.asarray([[event.token_id]]), jnp.asarray([[position]]), state=state,
                    return_updated_state=True,
                    forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
                    keychain=Keychain.init(0, sharding_config=model.sharding_config))
                assert decoded.updated_state is not None
                state, logits = decoded.updated_state, decoded.logits[:, 0]
        assert max_logprob_error < max_error, (model_name, max_logprob_error)
        assert max_greedy_gap < 0.1, (model_name, max_greedy_gap)
        stop_engine = ContinuousBatchingEngine(
            model, ContinuousBatchingConfig(page_size=page_size,
                                             total_pages=(prompt_lengths[0] + 2 + page_size - 1) // page_size,
                                             slot_count=1))
        stop_queue = stop_engine.submit(
            prompts[0], 2, GenerationConfig(stop_token_ids=(events[0][0].token_id,), temperature=0.0), 0)
        assert stop_engine.step() and stop_engine.step()
        assert stop_queue.get() == TokenEvent(finish_reason="stop", completion_tokens=1)
