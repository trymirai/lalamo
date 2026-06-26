from collections.abc import Generator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.model_import.loaders.dflash_loader import load_hf_dflash_draft_model
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.models.language_model import GenerationConfig
from lalamo.models.raw_text_codec import RawTextCodecConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.speculator import DFlashSpeculator, DFlashSpeculatorConfig, Speculator
from tests.conftest import ConvertModel, load_converted_model
from tests.generation.common import (
    sharded_generation_batch,
    stable_generation_forward_pass_configs,
    take_batch_prefix,
)

DFLASH_GENERATION_CASES = [
    (
        "Qwen/Qwen3-4B",
        "z-lab/Qwen3-4B-DFlash-b16",
        [
            [1, 1, 1, 2, 4, 1, 0, 0, 0, 0],
            [1, 1, 1, 2, 2, 1, 2, 0, 0, 0],
            [1, 1, 1, 2, 3, 2, 0, 0, 0, 0],
        ],
    ),
    # Enable Qwen/Qwen3.5-4B with z-lab/Qwen3.5-4B-DFlash after GDN rollback works.
]


@pytest.fixture(
    params=DFLASH_GENERATION_CASES,
    ids=[target_repo for target_repo, _, _ in DFLASH_GENERATION_CASES],
)
def dflash_generation_fixture(
    request: pytest.FixtureRequest,
    convert_model: ConvertModel,
) -> Generator[tuple[LanguageModel, DFlashSpeculator, list[list[int]]]]:
    target_repo, draft_repo, expected_step_lengths = request.param
    model_dir = convert_model(target_repo, cached=True)
    model = load_converted_model(model_dir, ShardingConfig.replicated())
    assert isinstance(model, LanguageModel)

    with jax.set_mesh(model.sharding_config.mesh):
        draft_model = load_hf_dflash_draft_model(
            snapshot_download(
                draft_repo,
                allow_patterns=["config.json", "*.safetensors"],
            ),
            sharding_config=model.sharding_config,
            dtype=jnp.bfloat16,
        )
        token_codec_config = RawTextCodecConfig()
        tokenizer = Tokenizer.from_str(dummy_char_level_tokenizer_config())
        yield (
            model,
            DFlashSpeculator(
                config=DFlashSpeculatorConfig(
                    token_codec_config=token_codec_config,
                    draft_config=draft_model.config,
                ),
                sharding_config=model.sharding_config,
                token_codec=token_codec_config.init(tokenizer),
                draft_model=draft_model,
            ),
            expected_step_lengths,
        )


def _generate(
    language_model: LanguageModel,
    speculator: Speculator | None,
    *,
    key_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prompts = [
        UserMessage("What's the capital of UK?"),
        UserMessage("Talk about apples"),
        UserMessage("Explain why the sky is blue"),
    ]
    encoded_prompts = [jnp.asarray(language_model.token_codec.encode_request([prompt])) for prompt in prompts]
    max_prompt_length = max(prompt.size for prompt in encoded_prompts)
    prompt_lengths = jnp.asarray([prompt.size for prompt in encoded_prompts], dtype=jnp.int32)
    token_ids = jnp.asarray(
        [jnp.pad(prompt, (0, max_prompt_length - prompt.size), constant_values=0) for prompt in encoded_prompts],
    )
    token_ids, prompt_lengths = sharded_generation_batch(language_model, token_ids, prompt_lengths)
    prefill_forward_pass_config, decode_forward_pass_config = stable_generation_forward_pass_configs()
    result = language_model.generate_tokens(
        token_ids,
        generation_config=GenerationConfig(temperature=0.0),
        prompt_lengths_without_padding=prompt_lengths,
        max_output_length=10,
        num_top_logits_to_return=8,
        prefill_forward_pass_config=prefill_forward_pass_config,
        decode_forward_pass_config=decode_forward_pass_config,
        keychain=Keychain.init(key_seed, sharding_config=language_model.sharding_config),
        speculator=speculator,
    )
    assert result.top_k_token_ids is not None
    assert result.top_k_token_logits is not None
    return (
        take_batch_prefix(language_model, result.token_ids, len(prompts)),
        take_batch_prefix(language_model, result.step_lengths, len(prompts)),
        take_batch_prefix(language_model, result.top_k_token_ids, len(prompts)),
        take_batch_prefix(language_model, result.top_k_token_logits, len(prompts)),
    )


def test_greedy_generation_matches_dflash_speculator(
    dflash_generation_fixture: tuple[LanguageModel, DFlashSpeculator, list[list[int]]],
) -> None:
    language_model, speculator, expected_step_lengths = dflash_generation_fixture

    ref_token_ids, _, ref_top_k_token_ids, ref_top_k_token_logits = _generate(
        language_model,
        None,
        key_seed=0,
    )
    spec_token_ids, spec_step_lengths, spec_top_k_token_ids, spec_top_k_token_logits = _generate(
        language_model,
        speculator,
        key_seed=0,
    )

    np.testing.assert_array_equal(spec_token_ids, ref_token_ids)
    np.testing.assert_array_equal(spec_step_lengths, expected_step_lengths)
    np.testing.assert_array_equal(spec_top_k_token_ids, ref_top_k_token_ids)
    np.testing.assert_allclose(spec_top_k_token_logits, ref_top_k_token_logits, atol=1e-4, rtol=1e-4)
