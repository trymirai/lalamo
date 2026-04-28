import json
from dataclasses import asdict
from typing import cast

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("torch")

import torch
from transformers import GenerationConfig as TransformersGenerationConfig
from transformers.generation.utils import GenerationMixin

from lalamo.model_import.common import FileSpec, download_file
from lalamo.model_import.huggingface_generation_config import HFGenerationConfig, _policy_from_hf_config
from lalamo.model_import.model_specs.common import ModelSpec, ModelType
from lalamo.models.language_model import GenerationConfig
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.sampling import (
    CompositePolicy,
    FrequencyPenalty,
    PresencePenalty,
    RepetitionPenalty,
)
from tests.common import assert_close
from tests.conftest import filter_specs, mark_by_size
from tests.model_test_tiers import ModelTier

standard_llm_specs = filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.STANDARD)


@pytest.mark.parametrize("spec", mark_by_size(standard_llm_specs), ids=[s.repo for s in standard_llm_specs])
def test_logit_processing(spec: ModelSpec) -> None:
    # TODO: lalamo should do greedy for do_sample=False
    generation_config = spec.configs.generation_config

    if isinstance(generation_config, GenerationConfig):
        generation_config_dict = asdict(generation_config)
        generation_config_dict.pop("stop_token_ids")
        generation_config_dict.pop("banned_tokens")
        generation_config_dict.pop("presence_penalty")
        generation_config_dict.pop("frequency_penalty")
        lalamo_hf_generation_config = HFGenerationConfig(**generation_config_dict)
        hf_generation_config = TransformersGenerationConfig.from_dict(
            {**asdict(lalamo_hf_generation_config), "do_sample": True},
        )
    elif isinstance(generation_config, FileSpec):
        hf_generation_config_file = download_file(generation_config, spec.repo)
        hf_generation_config_dict = json.loads(hf_generation_config_file.read_text())
        hf_generation_config = TransformersGenerationConfig.from_dict({**hf_generation_config_dict, "do_sample": True})
        lalamo_hf_generation_config = HFGenerationConfig.from_json(hf_generation_config_file)
    else:
        hf_generation_config = TransformersGenerationConfig(do_sample=True)
        lalamo_hf_generation_config = HFGenerationConfig()

    hf_processors = GenerationMixin()._get_logits_processor(hf_generation_config, input_ids_seq_length=1)  # noqa: SLF001
    lalamo_policy = (
        _policy_from_hf_config(lalamo_hf_generation_config)
        .default_policy(vocab_size=256)
        .init(
            prompt_token_ids=jnp.zeros((1,), dtype=jnp.int32),
            prompt_length=jnp.asarray(1, dtype=jnp.int32),
        )
    )

    for i in range(256):
        key = jax.random.PRNGKey(i)

        logits = jax.random.normal(key, (256,), dtype=jnp.float32)

        lalamo_result = lalamo_policy.process_logits(logits)

        hf_scores = cast("FloatTensor", torch.tensor(jax.device_get(logits), dtype=torch.float32).unsqueeze(0))
        hf_input_ids = cast("LongTensor", torch.zeros((1, 1), dtype=torch.long))
        hf_result = torch_to_jax(hf_processors(hf_input_ids, hf_scores)[0])

        assert_close(
            result=lalamo_result,
            reference=hf_result,
            atol=1e-6,
            rtol=1e-6,
            fraction_of_allowed_violations=0.01,
        )


def test_counting_penalty_stateful_update() -> None:
    vocab_size = 8
    prompt = jnp.asarray([1, 2, 1, 0, 0], dtype=jnp.int32)
    prompt_length = jnp.asarray(3, dtype=jnp.int32)

    policy = CompositePolicy(
        (
            RepetitionPenalty.zero(2.0, vocab_size),
            PresencePenalty.zero(0.5, vocab_size),
            FrequencyPenalty.zero(0.25, vocab_size),
        )
    ).init(prompt, prompt_length)

    policy = policy.update(jnp.asarray(3, dtype=jnp.int32))
    policy = policy.update(jnp.asarray(1, dtype=jnp.int32))

    expected_counts = jnp.asarray([0, 3, 1, 1, 0, 0, 0, 0], dtype=jnp.int32)
    for sub_policy in policy.policies:
        assert jnp.array_equal(sub_policy.token_counts, expected_counts)

    logits = jnp.asarray([1.0, 2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    counts = expected_counts.astype(jnp.float32)

    seen = counts > 0
    positive = logits > 0
    expected_rep = jnp.where(seen & positive, logits / 2.0, jnp.where(seen, logits * 2.0, logits))
    expected_presence = jnp.where(seen, expected_rep - 0.5, expected_rep)
    expected_final = expected_presence - 0.25 * counts

    assert_close(
        result=policy.process_logits(logits),
        reference=expected_final,
        atol=1e-6,
        rtol=1e-6,
        fraction_of_allowed_violations=0.0,
    )


def test_counting_penalty_ignores_out_of_vocab_prompt_tokens() -> None:
    vocab_size = 4
    prompt = jnp.asarray([1, 99, 2], dtype=jnp.int32)
    prompt_length = jnp.asarray(3, dtype=jnp.int32)

    policy = CompositePolicy((PresencePenalty.zero(1.0, vocab_size),)).init(prompt, prompt_length)

    (sub_policy,) = policy.policies
    assert jnp.array_equal(sub_policy.token_counts, jnp.asarray([0, 1, 1, 0], dtype=jnp.int32))
