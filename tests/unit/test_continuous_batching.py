import asyncio
from dataclasses import replace
from typing import Any, cast

import httpx2
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from openai import AsyncOpenAI, AsyncStream, BadRequestError, NotFoundError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel

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
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules import DecoderForwardPassConfig
from lalamo.modules.activations import SiLU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.mamba import Mamba2Config
from lalamo.server import create_app
from tests.helpers import build_tiny_attention_decoder_config, dense_log_softmax_rows

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def recurrent_model() -> LanguageModel:
    """A CPU model with only recurrent state, whose tokens decode to distinct words `w<id>.`."""
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
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "prompt": 1, **{f"w{index}.": index for index in range(2, 64)}}, unk_token="[UNK]")
    )
    tokenizer.decoder = Fuse()
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig("prompt", None, "system", "user", "assistant", None, None),
        decoder_config=decoder,
        generation_config=GenerationConfig(),
    )
    model = config.init(
        tokenizer,
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:1]),
            key=jax.random.key(7),
        ),
    )
    assert model.decoder.vocab_size <= 64
    return model


def _run(engine: ContinuousBatchingEngine) -> None:
    while engine.step():
        pass


def test_requests_match_dense_generation_after_slot_and_prefix_reuse(recurrent_model: LanguageModel) -> None:
    engine = ContinuousBatchingEngine(
        recurrent_model,
        ContinuousBatchingConfig(slot_count=2, max_context_length=48, prefill_batch_size=2, prefill_chunk_size=3),
    )
    first = tuple(range(1, 24))
    # The third prompt extends a cached prefix of the first and arrives after both slots are taken.
    prompts = [first, tuple(range(3, 21)), (*first[:15], 30)]
    budgets = [5, 9, 6]
    events: list[list[TokenEvent]] = [[], [], []]
    generation = GenerationConfig(temperature=0.0)
    for prompt, budget, received in zip(prompts[:2], budgets[:2], events[:2], strict=True):
        engine.submit(prompt, budget, generation, 0, return_logprobs=True, on_events=received.extend)
    assert engine.step()
    engine.submit(prompts[2], budgets[2], generation, 0, return_logprobs=True, on_events=events[2].extend)
    _run(engine)
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
            assert isinstance(event, GeneratedToken) and event.logprobs is not None
            np.testing.assert_allclose(event.logprobs.logprob, row[event.token_id], rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(
                event.logprobs.top_logprobs, row[jnp.asarray(event.logprobs.top_token_ids)], rtol=1e-4, atol=1e-3
            )


def test_signed_64_bit_seeds_replay_after_slot_reuse(recurrent_model: LanguageModel) -> None:
    engine = ContinuousBatchingEngine(
        recurrent_model,
        ContinuousBatchingConfig(slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=4),
    )
    seeds = (0, 2**32, -1, -(2**63), 2**63 - 1)
    runs: list[list[list[TokenEvent]]] = []
    for order in (seeds, seeds[::-1]):
        events: dict[int, list[TokenEvent]] = {seed: [] for seed in order}
        for seed in order:
            engine.submit((1, 2, 3, 4), 8, GenerationConfig(temperature=1.0), seed, on_events=events[seed].extend)
        _run(engine)
        runs.append([events[seed] for seed in seeds])
    original, replayed = runs
    assert original == replayed
    assert len({tuple(events) for events in original}) == len(seeds)


def test_context_without_positional_limit_must_be_configured(recurrent_model: LanguageModel) -> None:
    with pytest.raises(ValueError, match="requires max_context_length"):
        ContinuousBatchingEngine(recurrent_model, ContinuousBatchingConfig(slot_count=1))


def test_prefill_continuation_preserves_prefix_when_padding_exceeds_capacity() -> None:
    codec_config = ChatCodecConfig("", None, "system", "user", "assistant", None, None)
    config = LanguageModelConfig(
        token_codec_config=codec_config,
        decoder_config=build_tiny_attention_decoder_config((None,)),
        generation_config=GenerationConfig(),
    )
    sharding_config = ShardingConfig.replicated(jax.devices("cpu")[:1])
    model = config.init(
        Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        RandomInitializer(default_dtype=jnp.float32, sharding_config=sharding_config, key=jax.random.key(4)),
    )
    keychain = Keychain.init(0, sharding_config=sharding_config)
    forward_pass_config = DecoderForwardPassConfig.for_tracer_tests()
    tokens = jax.random.randint(jax.random.key(5), (2, 56), 0, model.decoder.vocab_size)
    prefix = model.prefill_tokens(
        tokens[:, :24], 64, jnp.array([24, 0]), forward_pass_config, chunk_size=24, keychain=keychain
    )
    head = model.prefill_tokens(
        jnp.stack([jnp.pad(tokens[0, 24:48], (0, 8)), tokens[1, :32]]),
        64,
        jnp.array([24, 32]),
        forward_pass_config,
        chunk_size=24,
        initial_state=prefix.state,
        prefix_lengths=jnp.array([24, 0]),
        keychain=keychain,
    )
    continued = model.prefill_tokens(
        jnp.stack([tokens[0, 48:56], tokens[1, 32:40]]),
        64,
        forward_pass_config=forward_pass_config,
        chunk_size=8,
        initial_state=head.state,
        prefix_lengths=jnp.array([48, 32]),
        keychain=keychain,
    )
    unchunked = model.prefill_tokens(
        tokens, 64, jnp.array([56, 40]), forward_pass_config, chunk_size=56, keychain=keychain
    )
    np.testing.assert_allclose(continued.last_token_logits, unchunked.last_token_logits, rtol=1e-4, atol=1e-5)


def test_openai_client_completions(recurrent_model: LanguageModel) -> None:
    codec = recurrent_model.token_codec
    greedy = recurrent_model.stream_tokens(
        jnp.asarray([1], dtype=jnp.int32),
        GenerationConfig(temperature=0.0),
        4,
        keychain=Keychain.init(0, sharding_config=recurrent_model.sharding_config),
    )
    token_texts = [codec.decode_tokens([token_id]) for token_id in map(int, greedy)]
    text = "".join(token_texts)

    async def run() -> None:
        api = create_app(
            recurrent_model,
            "test",
            ContinuousBatchingConfig(slot_count=2, max_context_length=32, prefill_batch_size=2),
        )
        async with (
            api.router.lifespan_context(api),
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=api)) as http,
        ):
            client = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=http)
            request: dict[str, Any] = {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 4,
                "temperature": 0,
            }

            async def complete(**overrides: Any) -> ChatCompletion:  # noqa: ANN401
                return cast("ChatCompletion", await client.chat.completions.create(**(request | overrides)))

            async def stream(**overrides: Any) -> list[ChatCompletionChunk]:  # noqa: ANN401
                chunks = await client.chat.completions.create(**(request | overrides), stream=True)
                return [chunk async for chunk in cast("AsyncStream[ChatCompletionChunk]", chunks)]

            completion, chunks = await asyncio.gather(
                complete(logprobs=True, top_logprobs=2), stream(stream_options={"include_usage": True})
            )
            (choice,) = completion.choices
            assert choice.message.content == text
            assert choice.finish_reason == "length"
            assert completion.usage is not None and completion.usage.total_tokens == 5
            assert choice.logprobs is not None and choice.logprobs.content is not None
            assert [entry.token for entry in choice.logprobs.content] == token_texts
            assert all(len(entry.top_logprobs) == 2 for entry in choice.logprobs.content)

            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == text
            assert chunks[-2].choices[0].finish_reason == "length"
            assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 4

            stopped = await complete(stop=token_texts[-1])
            assert stopped.choices[0].message.content == text[: text.find(token_texts[-1])]
            assert stopped.choices[0].finish_reason == "stop"

            with pytest.raises(NotFoundError):
                await complete(model="other")
            with pytest.raises(BadRequestError):
                await complete(response_format={"type": "json_object"})
            with pytest.raises(BadRequestError):
                await complete(tools=[{"type": "function", "function": {"name": "f"}}])
            assert (await http.get("http://test/health")).json() == {"status": "ok"}

    asyncio.run(run())
