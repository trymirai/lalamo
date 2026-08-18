import math
from collections.abc import Generator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.inference.continuous_batching import (
    CompletedRequest,
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    TokenizedRequest,
)
from lalamo.models import GenerationConfig, GenerationResults, LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.module import Keychain
from lalamo.modules.token_mixers.attention import Attention
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close
from tests.conftest import ConvertModel

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# The maintained single-B200 language-model set. The 80B Qwen3-Next float32 trace requires at least 512 GiB
# and continuous batching currently rejects model configurations that shard the batch axis.
MODEL_REPOS = (
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "google/gemma-3-1b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
    "LiquidAI/LFM2.5-350M",
    "cartesia-ai/Llamba-1B",
    "openai/gpt-oss-20b",
)

PAGE_SIZE = 16
MAX_OUTPUT_LENGTH = 3
TOP_K = 32
# Matches the LanguageModel.generate_tokens prefill chunk size, so that the dense reference and the paged
# engine run bit-identical prefill and only paged decode is under test.
PREFILL_CHUNK_SIZE = 512
# bf16 activations through the Pallas paged-attention kernel differ from the XLA dense kernels by a few ulps.
LOGIT_ATOL = 5e-2
LOGIT_RTOL = 3.5e-2

SENTENCES = (
    "The quick brown fox jumps over the lazy dog.",
    "Alice met Bob in Paris on a rainy Tuesday afternoon.",
    "They discussed the history of the Roman Empire at length.",
    "Coffee beans from Brazil were unusually expensive that year.",
    "The rules of chess reward patience more than aggression.",
    "A lighthouse keeper counted seventeen ships before dawn.",
    "Mountains in the north stayed white until late spring.",
    "The library closes early on the first Friday of every month.",
)


@pytest.fixture(params=MODEL_REPOS, ids=MODEL_REPOS)
def language_model(request: pytest.FixtureRequest, _convert_model_session: ConvertModel) -> Generator[LanguageModel]:
    model_dir = _convert_model_session(request.param, cached=True)
    model = LanguageModel.load(model_dir, sharding_config=ShardingConfig.replicated())
    assert isinstance(model, LanguageModel)
    yield model
    del model
    jax.clear_caches()


def _largest_sliding_window(model: LanguageModel) -> int:
    return max(
        (
            layer.mixer.config.sliding_window_size or 0
            for layer in model.decoder.transformer.layers
            if isinstance(layer.mixer, Attention)
        ),
        default=0,
    )


def _text_prompt(model: LanguageModel, request_index: int, prompt_length: int) -> tuple[int, ...]:
    # Real chat-formatted text (with BOS and template tokens) keeps the distributions peaked, unlike random
    # tokens, so greedy tokens are stable and dropping early tokens from a sliding window is clearly visible.
    rotated = (*SENTENCES[request_index:], *SENTENCES[:request_index])
    token_ids = model.token_codec.encode_request([UserMessage(" ".join(rotated * 64))])
    assert len(token_ids) >= prompt_length
    return tuple(int(token_id) for token_id in token_ids[:prompt_length])


def _pages_per_request(context_length: int) -> int:
    return 1 << (math.ceil(context_length / PAGE_SIZE) - 1).bit_length()


def _dense_prompt_length(model: LanguageModel, prompt_length: int, max_output_length: int) -> int:
    if any(isinstance(layer.mixer, Attention) and layer.mixer.has_sinks for layer in model.decoder.transformer.layers):
        return prompt_length
    state_capacity = PAGE_SIZE * _pages_per_request(prompt_length + max_output_length)
    padded_prompt_length = state_capacity - max_output_length - 1
    if padded_prompt_length < prompt_length:
        padded_prompt_length += state_capacity
    return padded_prompt_length


def _dense_reference(model: LanguageModel, request: TokenizedRequest, padded_length: int) -> GenerationResults:
    prompt_length = len(request.prompt_token_ids)
    prompt_token_ids = np.zeros((1, padded_length), dtype=np.int32)
    prompt_token_ids[0, :prompt_length] = request.prompt_token_ids
    return model.generate_tokens(
        jnp.asarray(prompt_token_ids),
        generation_config=request.generation_config,
        prompt_lengths_without_padding=jnp.asarray([prompt_length], dtype=jnp.int32),
        max_output_length=request.max_output_length,
        num_top_logits_to_return=request.num_top_logits,
        keychain=Keychain.init(request.seed, sharding_config=model.sharding_config),
    )


def _assert_matches_dense_reference(result: CompletedRequest, reference: GenerationResults) -> None:
    assert result.error is None
    assert result.logits is not None
    assert reference.top_k_token_ids is not None
    assert reference.top_k_token_logits is not None
    assert reference.remainder_logits is not None
    reference_token_ids = reference.token_ids[0].tolist()
    reference_top_token_ids = reference.top_k_token_ids[0]
    reference_top_logits = reference.top_k_token_logits[0]

    # Greedy is exact argmax on both paths, so kernel-level bf16 noise may legitimately flip a genuine
    # near-tie; the contexts diverge from that step on, so logits are compared through the first mismatch.
    mismatches = [
        step
        for step, (actual, expected) in enumerate(zip(result.output_token_ids, reference_token_ids, strict=False))
        if actual != expected
    ]
    if mismatches:
        (divergence_step, *_) = mismatches
        actual_token = result.output_token_ids[divergence_step]
        (actual_rank,) = np.flatnonzero(np.asarray(reference_top_token_ids[divergence_step]) == actual_token)
        expected_logit = float(reference_top_logits[divergence_step, 0])
        actual_logit = float(reference_top_logits[divergence_step, actual_rank])
        assert expected_logit - actual_logit <= 2 * (LOGIT_ATOL + LOGIT_RTOL * abs(expected_logit)), (
            f"paged greedy token {actual_token} diverged from dense {reference_token_ids[divergence_step]}"
            f" at step {divergence_step} without a near-tie ({expected_logit} vs {actual_logit})"
        )
        compared_steps = divergence_step + 1
    else:
        compared_steps = len(result.output_token_ids)

    compared_logits = result.logits[:compared_steps]
    actual_top_token_ids = jnp.asarray([step.top_token_ids for step in compared_logits], dtype=jnp.int32)
    actual_top_logits = jnp.asarray([step.top_raw_logits for step in compared_logits], dtype=jnp.float32)
    actual_remainder_logits = jnp.asarray([step.remainder_logit for step in compared_logits], dtype=jnp.float32)
    reference_top_token_ids = reference_top_token_ids[:compared_steps]
    reference_top_logits = reference_top_logits[:compared_steps]
    token_matches = actual_top_token_ids[:, :, None] == reference_top_token_ids[:, None, :]
    logit_tolerance = LOGIT_ATOL + LOGIT_RTOL * jnp.abs(reference_top_logits[:, None, :])
    matched_logit_residual = jnp.where(
        token_matches,
        jnp.abs(actual_top_logits[:, :, None] - reference_top_logits[:, None, :]) - logit_tolerance,
        -jnp.inf,
    )
    worst_step, worst_actual_rank, worst_reference_rank = np.unravel_index(
        int(jnp.argmax(matched_logit_residual)),
        matched_logit_residual.shape,
    )
    assert jnp.all(
        ~token_matches | (jnp.abs(actual_top_logits[:, :, None] - reference_top_logits[:, None, :]) <= logit_tolerance)
    ), (
        f"token {actual_top_token_ids[worst_step, worst_actual_rank]} at step {worst_step}: "
        f"paged {actual_top_logits[worst_step, worst_actual_rank]}, "
        f"dense {reference_top_logits[worst_step, worst_reference_rank]}, "
        f"tolerance {logit_tolerance[worst_step, 0, worst_reference_rank]}"
    )
    actual_missing = ~jnp.any(token_matches, axis=-1)
    reference_missing = ~jnp.any(token_matches, axis=-2)
    assert jnp.all(
        ~actual_missing | (actual_top_logits <= reference_top_logits[:, -1, None] + logit_tolerance[:, 0, -1, None])
    )
    assert jnp.all(
        ~reference_missing | (reference_top_logits <= actual_top_logits[:, -1, None] + logit_tolerance[:, 0, :])
    )
    assert_close(
        result=actual_top_logits,
        reference=reference_top_logits,
        atol=LOGIT_ATOL,
        rtol=LOGIT_RTOL,
        operation_name=f"{result.request_id} top-{TOP_K} raw logits",
    )
    assert_close(
        result=actual_remainder_logits,
        reference=reference.remainder_logits[0, :compared_steps],
        atol=LOGIT_ATOL,
        rtol=LOGIT_RTOL,
        operation_name=f"{result.request_id} remainder logits",
    )


def test_paged_greedy_and_top32_logits_match_across_arrivals_page_boundaries_and_sliding_windows(
    language_model: LanguageModel,
) -> None:
    # Prompts straddle page boundaries; sliding-window models additionally exceed the window by a wide margin
    # so that paged decode must drop BOS, the template and many words exactly like dense decode does.
    window_overflow = _largest_sliding_window(language_model) * 3 // 2
    prompt_lengths = tuple(base_length + window_overflow for base_length in (15, 15, 15, 15, 16, 17, 31, 33))
    requests = tuple(
        TokenizedRequest(
            request_id=f"request_{request_index}",
            sequence_id=f"sequence_{request_index}",
            prompt_token_ids=_text_prompt(language_model, request_index, prompt_length),
            max_output_length=MAX_OUTPUT_LENGTH,
            generation_config=GenerationConfig(temperature=0.0),
            seed=request_index,
            num_top_logits=TOP_K,
        )
        for request_index, prompt_length in enumerate(prompt_lengths)
    )
    references = {
        request.request_id: _dense_reference(
            language_model,
            request,
            _dense_prompt_length(language_model, len(request.prompt_token_ids), request.max_output_length),
        )
        for request in requests
    }

    engine = ContinuousBatchingEngine(
        language_model,
        ContinuousBatchingConfig(
            page_size=PAGE_SIZE,
            total_pages=4 * _pages_per_request(max(prompt_lengths) + MAX_OUTPUT_LENGTH),
            slot_count=4,
            max_decode_batch_size=4,
            prefill_batch_size=1,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            decode_steps_per_prefill=1,
        ),
    )
    for request in requests[:4]:
        engine.submit(request)
    assert engine.step()
    assert engine.step()
    assert engine.step()
    for request in requests[4:]:
        engine.submit(request)

    completed: dict[str, CompletedRequest] = {}
    for _ in range(128):
        assert engine.step()
        while result := engine.pop_completed():
            completed[result.request_id] = result
        if len(completed) == len(requests):
            break

    assert set(completed) == set(references)
    for request_id, reference in references.items():
        _assert_matches_dense_reference(completed[request_id], reference)
