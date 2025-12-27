import json
from dataclasses import asdict
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import FloatTensor, LongTensor
from transformers import GenerationConfig as TransformersGenerationConfig
from transformers.generation.utils import GenerationMixin

from lalamo.model_import.common import FileSpec, download_file
from lalamo.model_import.decoder_configs.huggingface import HuggingFaceLMConfig
from lalamo.model_import.huggingface_generation_config import HFGenerationConfig, _policy_from_hf_config
from lalamo.model_import.model_specs import ALL_MODELS
from lalamo.models.language_model import GenerationConfig
from lalamo.modules.torch_interop import torch_to_jax


@pytest.mark.parametrize(
    ["model_repo", "generation_config"],
    [(m.repo, m.configs.generation_config) for m in ALL_MODELS if issubclass(m.config_type, HuggingFaceLMConfig)],
    ids=[m.name for m in ALL_MODELS if issubclass(m.config_type, HuggingFaceLMConfig)],
)
def test_logit_processing(model_repo: str, generation_config: FileSpec | GenerationConfig | None) -> None:
    # TODO: lalamo should do greedy for do_sample=False
    if isinstance(generation_config, GenerationConfig):
        generation_config_dict = asdict(generation_config)
        generation_config_dict.pop("stop_token_ids")
        generation_config_dict.pop("banned_tokens")
        lalamo_hf_generation_config = HFGenerationConfig(**generation_config_dict)
        hf_generation_config = TransformersGenerationConfig.from_dict(
            {**asdict(lalamo_hf_generation_config), "do_sample": True},
        )
    elif isinstance(generation_config, FileSpec):
        hf_generation_config_file = download_file(generation_config, model_repo)
        hf_generation_config_dict = json.loads(hf_generation_config_file.read_text())
        hf_generation_config = TransformersGenerationConfig.from_dict({**hf_generation_config_dict, "do_sample": True})
        lalamo_hf_generation_config = HFGenerationConfig.from_json(hf_generation_config_file)
    else:
        hf_generation_config = TransformersGenerationConfig(do_sample=True)
        lalamo_hf_generation_config = HFGenerationConfig()

    hf_generation_config.repetition_penalty = None  # TODO: repetition penalty not implemented for now

    hf_processors = GenerationMixin()._get_logits_processor(hf_generation_config, input_ids_seq_length=1)  # noqa: SLF001
    lalamo_policy = _policy_from_hf_config(lalamo_hf_generation_config).default_policy()

    for i in range(256):
        key = jax.random.PRNGKey(i)

        logits = jax.random.normal(key, (256,), dtype=jnp.float32)

        lalamo_result = lalamo_policy.process_logits(logits)

        hf_scores = cast("FloatTensor", torch.tensor(jax.device_get(logits), dtype=torch.float32).unsqueeze(0))
        hf_input_ids = cast("LongTensor", torch.zeros((1, 1), dtype=torch.long))
        hf_result = torch_to_jax(hf_processors(hf_input_ids, hf_scores)[0])

        np.testing.assert_allclose(lalamo_result, hf_result)
