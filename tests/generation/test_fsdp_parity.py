from collections.abc import Callable
from random import Random

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer

from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.inference.batch_scheduler import (
    BatchSchedulerConfig,
    ContinuousBatchScheduler,
    FixedSizeBatchScheduler,
    GeneratedSequence,
)
from lalamo.initializer import RandomInitializer
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import GenerationConfig, LanguageModel, LanguageModelConfig, PrefillResults
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules import DecoderForwardPassConfig
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder


def _chat_codec_config() -> ChatCodecConfig:
    return ChatCodecConfig(
        prompt_template="",
        output_parser_regex=None,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
        end_of_thinking_tag=None,
    )


def _language_model(sharding_config: ShardingConfig) -> LanguageModel:
    decoder_config = build_tiny_attention_decoder((None,)).config
    config = LanguageModelConfig(
        token_codec_config=_chat_codec_config(),
        decoder_config=decoder_config,
        generation_config=GenerationConfig(temperature=0.0),
    )
    model = config.init(
        Tokenizer.from_str(dummy_char_level_tokenizer_config()),
        RandomInitializer(
            default_dtype=jnp.bfloat16,
            sharding_config=sharding_config,
            key=jax.random.key(0),
        ),
    )
    assert isinstance(model, LanguageModel)
    return model


def _batch_inputs(
    model: LanguageModel,
) -> tuple[jax.Array, jax.Array]:
    token_ids = jnp.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 0],
            [12, 13, 14, 15, 0, 0],
            [16, 17, 18, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    lengths = jnp.asarray([6, 5, 4, 3], dtype=jnp.int32)
    batch_axis = model.sharding_config.resolve_axis(LogicalAxis.BATCH)
    return (
        jax.device_put(token_ids, model.sharding_config.make_sharding((batch_axis, None))),
        jax.device_put(lengths, model.sharding_config.make_sharding((batch_axis,))),
    )


def _prefill(model: LanguageModel) -> PrefillResults:
    token_ids, lengths = _batch_inputs(model)
    return model.prefill_tokens(
        token_ids,
        state_capacity=12,
        lengths_without_padding=lengths,
        forward_pass_config=DecoderForwardPassConfig.for_inference(),
        keychain=Keychain.init(1, shape=(token_ids.shape[0],), sharding_config=model.sharding_config),
    )


def _decode_logits(model: LanguageModel, prefill: PrefillResults) -> jax.Array:
    next_token_ids = jnp.argmax(prefill.last_token_logits, axis=-1).astype(jnp.int32)
    batch_axis = model.sharding_config.resolve_axis(LogicalAxis.BATCH)
    next_token_ids = jax.device_put(next_token_ids, model.sharding_config.make_sharding((batch_axis,)))
    result = model.decoder(
        token_ids=next_token_ids[:, None],
        token_positions=(prefill.last_token_indices + 1)[:, None],
        state=prefill.state,
        return_updated_state=True,
        forward_pass_config=DecoderForwardPassConfig.for_inference(),
        keychain=Keychain.init(2, shape=(next_token_ids.shape[0],), sharding_config=model.sharding_config),
    )
    return result.logits[:, 0, :]


@pytest.mark.parametrize(
    ("parallelism", "sharding_config"),
    [
        ("FSDP", ShardingConfig.fully_sharded_data_parallel),
        ("tensor parallel", ShardingConfig.tensor_parallel),
    ],
)
def test_model_parallel_logits_match_data_parallel(
    parallelism: str,
    sharding_config: Callable[[list[jax.Device]], ShardingConfig],
) -> None:
    devices = jax.devices("cpu")[:4]
    single_device_model = _language_model(ShardingConfig.replicated(devices[:1]))
    data_parallel_model = _language_model(ShardingConfig.data_parallel(devices))
    model_parallel = _language_model(sharding_config(devices))

    data_parallel_prefill = _prefill(data_parallel_model)
    model_parallel_prefill = _prefill(model_parallel)
    token_ids, lengths = _batch_inputs(single_device_model)
    single_request_logits = []
    for row, length in enumerate(np.asarray(lengths)):
        result = single_device_model.prefill_tokens(
            token_ids[row : row + 1, :length],
            state_capacity=12,
            keychain=Keychain.init(row + 3, sharding_config=single_device_model.sharding_config),
        )
        single_request_logits.append(result.last_token_logits[0])

    assert_close(
        result=model_parallel_prefill.last_token_logits,
        reference=data_parallel_prefill.last_token_logits,
        operation_name=f"{parallelism} prefill logits",
    )
    assert_close(
        result=model_parallel_prefill.last_token_logits,
        reference=jnp.stack(single_request_logits),
        operation_name=f"{parallelism} one-by-one prefill logits",
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.argmax(model_parallel_prefill.last_token_logits, axis=-1)),
        np.asarray(jnp.argmax(data_parallel_prefill.last_token_logits, axis=-1)),
    )
    assert_close(
        result=_decode_logits(model_parallel, model_parallel_prefill),
        reference=_decode_logits(data_parallel_model, data_parallel_prefill),
        operation_name=f"{parallelism} decode logits",
    )


@pytest.mark.parametrize(
    "sharding_config",
    [
        pytest.param(ShardingConfig.replicated, id="normal"),
        pytest.param(ShardingConfig.fully_sharded_data_parallel, id="fsdp"),
        pytest.param(ShardingConfig.tensor_parallel, id="tensor-parallel"),
    ],
)
def test_continuous_batching_matches_fixed_for_random_inputs(
    sharding_config: Callable[[list[jax.Device]], ShardingConfig],
) -> None:
    devices = jax.devices("cpu")[:4]
    model_parallel = _language_model(sharding_config(devices))
    rng = Random(0)
    generation_config = GenerationConfig(temperature=0.0)

    for padded_length in (8, rng.randint(9, 64), rng.randint(65, 128)):
        batch_size = rng.choice((4, 8))
        max_output_length = rng.randint(1, 6)
        tokenized = [
            [rng.randrange(1, 32) for _ in range(rng.randint(1, padded_length))]
            for _ in range(rng.randint(1, batch_size * 2))
        ]
        batch_scheduler_config = BatchSchedulerConfig(
            batch_size=batch_size,
            max_output_length=max_output_length,
            padded_length=padded_length,
        )
        expected = dict(
            FixedSizeBatchScheduler(model=model_parallel).generate_tokens_many(
                tokenized,
                generation_config=generation_config,
                batch_scheduler_config=batch_scheduler_config,
            ),
        )
        actual = dict(
            ContinuousBatchScheduler(
                model=model_parallel,
                block_size=rng.randint(1, max_output_length),
            ).generate_tokens_many(
                tokenized,
                generation_config=generation_config,
                batch_scheduler_config=batch_scheduler_config,
            ),
        )

        assert actual.keys() == expected.keys() == set(range(len(tokenized)))
        for sequence_id, expected_result in expected.items():
            np.testing.assert_array_equal(actual[sequence_id].token_ids, expected_result.token_ids)


def test_continuous_batching_matches_normal_for_partial_prefill_chunks() -> None:
    devices = jax.devices("cpu")[:4]
    rng = Random(1)
    tokenized = [[rng.randrange(1, 32) for _ in range(rng.randint(1, 150))] for _ in range(rng.randint(1, 8))]
    batch_scheduler_config = BatchSchedulerConfig(batch_size=4, max_output_length=2, padded_length=150)
    generation_config = GenerationConfig(temperature=0.0)

    def generate(sharding_config: ShardingConfig) -> dict[int, GeneratedSequence]:
        model = _language_model(sharding_config)
        return dict(
            ContinuousBatchScheduler(model=model, block_size=2).generate_tokens_many(
                tokenized,
                generation_config=generation_config,
                batch_scheduler_config=batch_scheduler_config,
            ),
        )

    expected = generate(ShardingConfig.replicated(devices))
    for sharding_config in (
        ShardingConfig.fully_sharded_data_parallel(devices),
        ShardingConfig.tensor_parallel(devices),
    ):
        actual = generate(sharding_config)
        assert actual.keys() == expected.keys()
        for sequence_id, expected_result in expected.items():
            np.testing.assert_array_equal(actual[sequence_id].token_ids, expected_result.token_ids)
