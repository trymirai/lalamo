from pathlib import Path

import jax
import jax.numpy as jnp

from lalamo.inference.batch_scheduler import (
    BatchScheduler,
    BatchSchedulerConfig,
    ContinuousBatchScheduler,
    FixedSizeBatchScheduler,
)
from lalamo.models.language_model import GenerationConfig, LanguageModel
from lalamo.module import Keychain
from lalamo.speculator import DFlashSpeculator, Speculator
from lalamo.utils.sharding import ShardingConfig
from tests.unit.speculator.conftest import VOCAB_SIZE

GREEDY_CONFIG = GenerationConfig(temperature=0.0)


def make_ragged_batch() -> tuple[jax.Array, jax.Array]:
    lengths = [5, 12, 9]
    max_length = max(lengths)
    token_ids = jnp.stack(
        [
            jnp.pad(jnp.arange(3 * index, 3 * index + length) % (VOCAB_SIZE - 1), (0, max_length - length))
            for index, length in enumerate(lengths)
        ],
    ).astype(jnp.int32)
    return token_ids, jnp.asarray(lengths, dtype=jnp.int32)


def generate(
    language_model: LanguageModel,
    speculator: Speculator | None,
    max_output_length: int = 16,
) -> jax.Array:
    token_ids, lengths = make_ragged_batch()
    with jax.set_mesh(language_model.sharding_config.mesh):
        results = language_model.generate_tokens(
            token_ids,
            generation_config=GREEDY_CONFIG,
            prompt_lengths_without_padding=lengths,
            max_output_length=max_output_length,
            keychain=Keychain.init(0, sharding_config=language_model.sharding_config),
            speculator=speculator,
        )
    return results.token_ids


def test_dflash_matches_baseline(language_model: LanguageModel, dflash_speculator: DFlashSpeculator) -> None:
    baseline_token_ids = generate(language_model, speculator=None)
    speculative_token_ids = generate(language_model, dflash_speculator)

    assert jnp.array_equal(baseline_token_ids, speculative_token_ids)


def test_bonus_only_step_matches_baseline(
    language_model: LanguageModel,
    dflash_speculator: DFlashSpeculator,
) -> None:
    baseline_token_ids = generate(language_model, speculator=None, max_output_length=1)
    speculative_token_ids = generate(language_model, dflash_speculator, max_output_length=1)

    assert jnp.array_equal(baseline_token_ids, speculative_token_ids)


def test_stream_is_prefix_of_generate(language_model: LanguageModel, dflash_speculator: DFlashSpeculator) -> None:
    token_ids, lengths = make_ragged_batch()
    prompt_token_ids = token_ids[1, : lengths[1]]
    with jax.set_mesh(language_model.sharding_config.mesh):
        reference = language_model.generate_tokens(
            prompt_token_ids[None, :],
            generation_config=GREEDY_CONFIG,
            max_output_length=8,
            keychain=Keychain.init(0, sharding_config=language_model.sharding_config),
            speculator=dflash_speculator,
        )
        streamed = [
            int(token_id.item())
            for token_id in language_model.stream_tokens(
                prompt_token_ids,
                generation_config=GREEDY_CONFIG,
                max_output_length=8,
                keychain=Keychain.init(0, sharding_config=language_model.sharding_config),
                speculator=dflash_speculator,
            )
        ]

    assert len(streamed) > 0
    assert streamed == reference.token_ids[0, : len(streamed)].tolist()


def test_schedulers_greedy_parity(language_model: LanguageModel, dflash_speculator: DFlashSpeculator) -> None:
    prompts = [
        [int(token) for token in jnp.arange(start, start + length) % (VOCAB_SIZE - 1)]
        for start, length in ((0, 5), (3, 12), (7, 9), (1, 15), (11, 6))
    ]
    scheduler_config = BatchSchedulerConfig(batch_size=2, max_output_length=8, padded_length=16)
    schedulers: dict[str, BatchScheduler] = {
        "fixed/none": FixedSizeBatchScheduler(model=language_model),
        "fixed/dflash": FixedSizeBatchScheduler(model=language_model, speculator=dflash_speculator),
        "continuous/none": ContinuousBatchScheduler(model=language_model, block_size=4),
        "continuous/dflash": ContinuousBatchScheduler(
            model=language_model,
            speculator=dflash_speculator,
            block_size=4,
        ),
        "continuous/dflash/chunked-prefill": ContinuousBatchScheduler(
            model=language_model,
            speculator=dflash_speculator,
            block_size=4,
            prefill_chunk_size=4,
        ),
    }

    with jax.set_mesh(language_model.sharding_config.mesh):
        results = {
            scheduler_name: dict(
                scheduler.generate_tokens_many(
                    prompts,
                    generation_config=GREEDY_CONFIG,
                    batch_scheduler_config=scheduler_config,
                    keychain=Keychain.init(0, shape=(len(prompts),), sharding_config=language_model.sharding_config),
                ),
            )
            for scheduler_name, scheduler in schedulers.items()
        }

    reference = results["fixed/none"]
    assert sorted(reference) == list(range(len(prompts)))
    for scheduler_name, result in results.items():
        assert sorted(result) == sorted(reference), scheduler_name
        for request_index in reference:
            assert jnp.array_equal(
                result[request_index].token_ids,
                reference[request_index].token_ids,
            ), f"{scheduler_name}, request {request_index}"


def test_save_load_roundtrip(
    tmp_path: Path,
    sharding_config: ShardingConfig,
    dflash_speculator: DFlashSpeculator,
) -> None:
    dflash_speculator.save(tmp_path)

    for file_name in ("config.json", "model.safetensors", "tokenizer.json"):
        assert (tmp_path / file_name).exists()

    loaded = Speculator.load(tmp_path, sharding_config)

    assert isinstance(loaded, DFlashSpeculator)
    assert loaded.config == dflash_speculator.config
    loaded_arrays = [leaf for leaf in jax.tree.leaves(loaded) if isinstance(leaf, jax.Array)]
    original_arrays = [leaf for leaf in jax.tree.leaves(dflash_speculator) if isinstance(leaf, jax.Array)]
    for loaded_array, original_array in zip(loaded_arrays, original_arrays, strict=True):
        assert jnp.array_equal(loaded_array, original_array)
