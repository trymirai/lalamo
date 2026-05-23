from collections.abc import Callable
from dataclasses import asdict
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
import torch
from transformers import GenerationConfig as TransformersGenerationConfig
from transformers.generation.utils import GenerationMixin

from lalamo.model_import.huggingface_generation_config import HFGenerationConfig, _policy_from_hf_config
from lalamo.module import Keychain
from lalamo.sampling import SamplingPolicy
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close
from tests.helpers import make_test_sharding_config


def _assert_array(result: jax.Array | None, expected: jax.Array) -> None:
    assert result is not None
    assert jnp.array_equal(result, expected)


def _with_counts(
    policy: SamplingPolicy,
    tokens: tuple[int, ...],
    length: int,
    vocabulary_size: int,
) -> SamplingPolicy:
    return policy.with_prompt_token_counts(
        jnp.array(tokens, dtype=jnp.int32),
        jnp.array(length, dtype=jnp.int32),
        vocabulary_size=vocabulary_size,
    )


@pytest.mark.parametrize(
    "hf_generation_config",
    [
        pytest.param(HFGenerationConfig(), id="default"),
        pytest.param(HFGenerationConfig(temperature=0.7), id="temperature"),
        pytest.param(HFGenerationConfig(top_k=50), id="top-k"),
        pytest.param(HFGenerationConfig(top_p=0.8), id="top-p"),
        pytest.param(HFGenerationConfig(min_p=0.15), id="min-p"),
        pytest.param(HFGenerationConfig(repetition_penalty=1.1), id="repetition-penalty"),
        pytest.param(HFGenerationConfig(temperature=0.3, min_p=0.15), id="temperature-min-p"),
        pytest.param(HFGenerationConfig(temperature=0.1, top_k=50, top_p=0.1), id="temperature-top-k-top-p"),
    ],
)
def test_process_logits_matches_huggingface_generation_config(hf_generation_config: HFGenerationConfig) -> None:
    transformers_config = TransformersGenerationConfig.from_dict(
        {**asdict(hf_generation_config), "do_sample": True},
    )
    hf_processors = cast("Any", GenerationMixin())._get_logits_processor(  # noqa: SLF001
        transformers_config,
        input_ids_seq_length=1,
    )
    lalamo_policy = (
        _policy_from_hf_config(hf_generation_config)
        .default_policy()
        .with_prompt_token_counts(
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            vocabulary_size=256,
        )
    )

    for i in range(256):
        logits = jax.random.normal(jax.random.key(i), (256,), dtype=jnp.float32)

        lalamo_result = lalamo_policy.process_logits(logits)

        hf_scores = cast("torch.FloatTensor", torch.tensor(jax.device_get(logits), dtype=torch.float32).unsqueeze(0))
        hf_input_ids = cast("torch.LongTensor", torch.zeros((1, 1), dtype=torch.long))
        hf_result = torch_to_jax(hf_processors(hf_input_ids, hf_scores)[0])

        assert_close(
            result=lalamo_result,
            reference=hf_result,
            atol=1e-6,
            rtol=1e-6,
            fraction_of_allowed_violations=0.01,
            operation_name=f"{hf_generation_config} seed={i}",
        )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SamplingPolicy.init(banned_tokens=range(17)), "At most 16 banned tokens"),
        (lambda: SamplingPolicy.init(banned_tokens=(-1,)), "Banned tokens must be non-negative"),
        (lambda: SamplingPolicy.init(repetition_penalty=0.0), "repetition_penalty must be positive"),
        (
            lambda: SamplingPolicy.init_batch(
                temperature=(0.5,),
                top_k=(1, 2),
            ),
            "same length",
        ),
    ],
)
def test_init_rejects_invalid_arguments(call: Callable[[], SamplingPolicy], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    ("policy", "logits", "expected"),
    [
        (SamplingPolicy.init(), [1.0, -2.0, 3.5], [1.0, -2.0, 3.5]),
        (SamplingPolicy.init(banned_tokens=(1, 3)), [0.0, 1.0, 2.0, 3.0], [0.0, -jnp.inf, 2.0, -jnp.inf]),
        (SamplingPolicy.init(temperature=0.0), [0.0, 3.0, 2.0], [-jnp.inf, 1.0, -jnp.inf]),
        (SamplingPolicy.init(temperature=2.0), [1.0, 2.0, 4.0], [0.5, 1.0, 2.0]),
        (SamplingPolicy.init(top_k=2), [1.0, 5.0, 3.0, 4.0], [-jnp.inf, 5.0, -jnp.inf, 4.0]),
        (SamplingPolicy.init(top_k=8), [1.0, 5.0, 3.0], [1.0, 5.0, 3.0]),
        (SamplingPolicy.init(top_k=1, top_p=0.5), [3.0, 3.0, 1.0], [3.0, -jnp.inf, -jnp.inf]),
        (SamplingPolicy.init(top_p=0.5), [5.0, 4.0, 3.0], [5.0, -jnp.inf, -jnp.inf]),
        (SamplingPolicy.init(top_p=0.9), [3.0, 2.0, 1.0], [3.0, 2.0, -jnp.inf]),
        (SamplingPolicy.init(top_p=0.5), [0.0, 0.0], [0.0, -jnp.inf]),
        (SamplingPolicy.init(min_p=0.2), jnp.log(jnp.array([1.0, 0.25, 0.05])), [0.0, -1.3862944, -jnp.inf]),
        (
            _with_counts(SamplingPolicy.init(repetition_penalty=2.0), (0, 1), 2, 4),
            [4.0, -3.0, 8.0, 1.0],
            [2.0, -6.0, 8.0, 1.0],
        ),
        (
            _with_counts(SamplingPolicy.init(presence_penalty=0.5), (0, 0, 2), 3, 3),
            [4.0, 3.0, 2.0],
            [3.5, 3.0, 1.5],
        ),
        (
            _with_counts(SamplingPolicy.init(frequency_penalty=0.5), (0, 0, 2), 3, 3),
            [4.0, 3.0, 2.0],
            [3.0, 3.0, 1.5],
        ),
        (SamplingPolicy.init(top_k=1, banned_tokens=(1,)), [1.0, 4.0, 3.0], [-jnp.inf, -jnp.inf, 3.0]),
    ],
)
def test_process_logits(policy: SamplingPolicy, logits: jax.Array | list[float], expected: list[float]) -> None:
    result = policy.process_logits(jnp.asarray(logits, dtype=jnp.float32))

    _assert_array(result, jnp.array(expected, dtype=jnp.float32))


def test_top_k_keeps_exactly_k_tokens_at_cutoff_ties() -> None:
    result = SamplingPolicy.init(top_k=2).process_logits(jnp.array([5.0, 4.0, 4.0, 4.0, 3.0], dtype=jnp.float32))

    assert jnp.isfinite(result).sum().item() == 2
    assert result[0].item() == 5.0


def test_token_counts_ignore_out_of_vocab_tokens_and_update_generated_tokens() -> None:
    policy = _with_counts(SamplingPolicy.init(repetition_penalty=2.0), (1, -1, 9, 2), 4, 4)
    updated_policy = policy.with_next_token_count(jnp.array(2, dtype=jnp.int32))
    updated_policy = updated_policy.with_next_token_count(jnp.array(99, dtype=jnp.int32))
    updated_policy = updated_policy.with_next_token_count(jnp.array(-1, dtype=jnp.int32))

    _assert_array(policy.token_counts, jnp.array([0, 1, 1, 0], dtype=jnp.int32))
    _assert_array(updated_policy.token_counts, jnp.array([0, 1, 2, 0], dtype=jnp.int32))
    _assert_array(
        policy.process_logits(jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)),
        jnp.array([1.0, 1.0, 2.0, 8.0], dtype=jnp.float32),
    )


def test_batched_policy_requires_vmap_and_processes_rows() -> None:
    policy = SamplingPolicy.init_batch(
        temperature=(0.0, 1.0),
        top_k=(0, 1),
        top_p=(1.0, 1.0),
        min_p=(0.0, 0.0),
        banned_tokens=((), ()),
    )
    logits = jnp.array([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0]], dtype=jnp.float32)

    with pytest.raises(ValueError, match="Use vmap"):
        policy.process_logits(logits[0])

    result = jax.vmap(lambda policy_row, logits_row: policy_row.process_logits(logits_row))(policy, logits)

    _assert_array(result, jnp.array([[-jnp.inf, 1.0, -jnp.inf], [-jnp.inf, 3.0, -jnp.inf]], dtype=jnp.float32))


def test_call_samples_greedy_token_when_temperature_is_zero() -> None:
    result = SamplingPolicy.init(temperature=0.0)(
        jnp.array([0.0, 3.0, 2.0], dtype=jnp.float32),
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )

    assert result.shape == ()
    assert result.item() == 1
