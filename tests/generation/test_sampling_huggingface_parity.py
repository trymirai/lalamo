from dataclasses import asdict
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
import torch
from transformers import GenerationConfig as TransformersGenerationConfig
from transformers.generation.utils import GenerationMixin

from lalamo.model_import.huggingface_generation_config import HFGenerationConfig, _policy_from_hf_config
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close


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
