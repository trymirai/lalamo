import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.inference.batch_scheduler import (
    BatchSchedulerConfig,
    ContinuousBatchScheduler,
    FixedSizeBatchScheduler,
)
from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    GeneratedToken,
    SequenceFinished,
    TokenEvent,
)
from lalamo.initializer import RandomInitializer
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig, UserMessage
from lalamo.models.language_model import GenerationConfig, LanguageModelConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules import DecoderForwardPassConfig
from tests.conftest import ConvertModel
from tests.helpers import build_tiny_attention_decoder_config

_FUZZ_MODEL_REPOS = (
    "Qwen/Qwen3-0.6B",
    "cartesia-ai/Llamba-1B",
)

_FUZZ_PROMPTS = (
    "Say hi.",
    "Name a fruit.",
    "What is 2+2?",
    "Reply with one word.",
    "Yes or no?",
    "Pick a color.",
    "Complete: the sky is",
    "One short word please.",
)


@pytest.fixture(scope="module", params=_FUZZ_MODEL_REPOS, ids=_FUZZ_MODEL_REPOS)
def fuzz_language_model(request: pytest.FixtureRequest, _convert_model_session: ConvertModel) -> LanguageModel:
    model_dir = _convert_model_session(request.param, cached=True)
    model = LanguageModel.load(model_dir, sharding_config=ShardingConfig.replicated())
    assert isinstance(model, LanguageModel)
    return model


@pytest.mark.parametrize(
    ("seed", "num_prompts", "batch_size", "block_size", "max_output_length", "padded_length"),
    [
        # (empty)                            zero sequences
        (0, 0, 2, 8, 8, 32),
        # (under-filled batch, misaligned)   10%8=2, lines stay empty forever
        (1, 1, 4, 8, 10, 48),
        # (exact fit, misaligned)            12%8=4, no refill
        (2, 4, 4, 8, 12, 56),
        # (heavy refill, misaligned)         14%4=2, many churns
        (3, 12, 2, 4, 14, 56),
        # (block_size=1)                     every step is a boundary
        (4, 3, 2, 1, 8, 64),
        # (block > max_output)               block clamps to max_output
        (5, 2, 2, 64, 6, 56),
        # (small batches, misaligned)        batch_size=2, 11%4=3
        (6, 5, 2, 4, 11, 40),
        # (multi-block, misaligned)          batch_size=4, 13%4=1, minimal tail
        (7, 8, 4, 4, 13, 80),
    ],
)
def test_continuous_vs_fixed_fuzz(
    fuzz_language_model: LanguageModel,
    seed: int,
    num_prompts: int,
    batch_size: int,
    block_size: int,
    max_output_length: int,
    padded_length: int,
) -> None:
    rng = random.Random(seed)
    prompts = [[UserMessage(rng.choice(_FUZZ_PROMPTS))] for _ in range(num_prompts)]
    tokenized = [fuzz_language_model.token_codec.encode_request(prompt) for prompt in prompts]

    generation_config = GenerationConfig(
        temperature=0.0,
        frequency_penalty=0.5,
        stop_token_ids=fuzz_language_model.config.generation_config.stop_token_ids,
    )
    batch_scheduler_config = BatchSchedulerConfig(
        batch_size=batch_size,
        max_output_length=max_output_length,
        padded_length=padded_length,
    )

    fixed_results = dict(
        FixedSizeBatchScheduler(model=fuzz_language_model).generate_tokens_many(
            tokenized,
            generation_config=generation_config,
            batch_scheduler_config=batch_scheduler_config,
        ),
    )
    continuous_results = dict(
        ContinuousBatchScheduler(
            model=fuzz_language_model,
            block_size=block_size,
        ).generate_tokens_many(
            tokenized,
            generation_config=generation_config,
            batch_scheduler_config=batch_scheduler_config,
        ),
    )

    assert fixed_results.keys() == continuous_results.keys() == set(range(num_prompts))
    for seq_id in range(num_prompts):
        fixed_ids = fuzz_language_model.trim_at_eos(fixed_results[seq_id].token_ids.tolist())
        continuous_ids = fuzz_language_model.trim_at_eos(continuous_results[seq_id].token_ids.tolist())
        assert fixed_ids == continuous_ids, f"seq {seq_id}: fixed={fixed_ids[:20]} continuous={continuous_ids[:20]}"


@pytest.fixture(scope="module")
def paged_language_model(_convert_model_session: ConvertModel) -> LanguageModel:
    return LanguageModel.load(
        _convert_model_session("Qwen/Qwen3.5-0.8B", cached=True), sharding_config=ShardingConfig.replicated()
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("cancel_active", [False, True], ids=["before_admission", "after_prefill"])
def test_engine_cancellation_allows_new_request(paged_language_model: LanguageModel, cancel_active: bool) -> None:
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


@pytest.mark.fast
def test_prefill_continuation_preserves_prefix_when_padding_exceeds_capacity() -> None:
    codec_config = ChatCodecConfig(
        prompt_template="",
        output_parser_regex=None,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
    )
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
