import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.compressed.low_rank import LowRankMatrix
from lalamo.distillation import AdvantageBaseline, partition_adapters
from lalamo.distillation.runner import (
    DistillationConfig,
    ExecutionConfig,
    PromptPair,
    StepReport,
    run_on_policy_distillation,
)
from lalamo.initializer import RandomInitializer
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import ShardingConfig
from lalamo.utils.surgery import select_nodes_of_type
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder_config

CONFIG = DistillationConfig(
    num_steps=3,
    prompts_per_step=4,
    max_response_length=6,
    adapter_rank=2,
    learning_rate=1e-2,
    gradient_clip_norm=1.0,
    baseline=AdvantageBaseline.BATCH_MEAN,
    seed=0,
)
EXECUTION = ExecutionConfig(
    gradient_micro_batch_size=2,
    student_scoring_micro_batch_size=4,
    teacher_scoring_micro_batch_size=2,
    checkpoint_every_steps=1,
)
# The teacher sees every conversation behind a longer prompt, as it does when its chat template adds a system line.
PROMPT_PAIRS = tuple(
    PromptPair(student=(5 + index % 7, 9), teacher=(11, 12, 5 + index % 7, 9))
    for index in range(CONFIG.num_steps * CONFIG.prompts_per_step)
)


def tiny_language_model(seed: int) -> LanguageModel:
    decoder_config = build_tiny_attention_decoder_config((None, None))
    tokenizer = Tokenizer(
        WordLevel({f"w{index}": index for index in range(decoder_config.vocab_size)}, unk_token="w0"),
    )
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig("prompt", None, "system", "user", "assistant", None, None),
        decoder_config=decoder_config,
        generation_config=GenerationConfig(stop_token_ids=(3,)),
    )
    return config.init(
        tokenizer,
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:1]),
            key=jax.random.key(seed),
        ),
    )


def adapter_weights(model: LanguageModel) -> list[Array]:
    adapters, _ = partition_adapters(model.decoder)
    return jax.tree.leaves(adapters)


def test_interrupted_run_resumes_to_the_adapters_of_an_uninterrupted_run(tmp_path: Path) -> None:
    student, teacher = tiny_language_model(seed=0), tiny_language_model(seed=1)
    reports: list[StepReport] = []
    uninterrupted = run_on_policy_distillation(
        student,
        teacher,
        PROMPT_PAIRS,
        CONFIG,
        EXECUTION,
        tmp_path / "uninterrupted",
        reports.append,
    )

    interrupted_directory = tmp_path / "interrupted"
    for num_steps in (2, CONFIG.num_steps):
        resumed = run_on_policy_distillation(
            student,
            teacher,
            PROMPT_PAIRS,
            replace(CONFIG, num_steps=num_steps),
            EXECUTION,
            interrupted_directory,
            lambda _: None,
        )

    assert [report.metrics.step for report in reports] == [0, 1, 2]
    assert all(math.isfinite(report.metrics.reverse_kl) for report in reports)
    assert sorted(path.name for path in (interrupted_directory / "checkpoints").iterdir()) == [
        "step_00001.eqx",
        "step_00002.eqx",
        "step_00003.eqx",
    ]
    assert len((interrupted_directory / "metrics.jsonl").read_text().splitlines()) == CONFIG.num_steps
    assert any(
        jnp.any(adapter.up_projection != 0)
        for _, adapter in select_nodes_of_type(LowRankMatrix, uninterrupted.decoder)
    )
    for resumed_weights, reference_weights in zip(
        adapter_weights(resumed), adapter_weights(uninterrupted), strict=True
    ):
        assert_close(result=resumed_weights, reference=reference_weights)
