import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.compressed.low_rank import LowRankMatrix
from lalamo.distillation import AdvantageBaseline, Rollouts, partition_adapters, sampled_token_log_probs
from lalamo.distillation.runner import (
    ExecutionConfig,
    OnPolicyConfig,
    PromptPair,
    StepReport,
    SupervisedConfig,
    SupervisedExample,
    SupervisedStepReport,
    run_on_policy_training,
    run_supervised_fine_tuning,
)
from lalamo.initializer import RandomInitializer
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules import DecoderForwardPassConfig
from lalamo.utils.surgery import select_nodes_of_type
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder_config

CONFIG = OnPolicyConfig(
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


def first_token_probabilities(model: LanguageModel, prompt: Sequence[int]) -> Array:
    """Probability of every token as the first token of a response to `prompt`."""
    vocab_size = model.decoder.vocab_size
    rollouts = Rollouts(
        token_ids=jnp.asarray([[*prompt, token_id] for token_id in range(vocab_size)], dtype=jnp.int32),
        prompt_lengths=jnp.full(vocab_size, len(prompt), dtype=jnp.int32),
        response_lengths=jnp.ones(vocab_size, dtype=jnp.int32),
    )
    log_probs = sampled_token_log_probs(
        model.decoder,
        rollouts,
        DecoderForwardPassConfig.for_inference(),
        keychain=Keychain.init(0, sharding_config=model.sharding_config),
    )
    return jnp.exp(log_probs[:, len(prompt) - 1])


def test_interrupted_run_resumes_to_the_adapters_of_an_uninterrupted_run(tmp_path: Path) -> None:
    student, teacher = tiny_language_model(seed=0), tiny_language_model(seed=1)
    reports: list[StepReport] = []
    uninterrupted = run_on_policy_training(
        student,
        teacher,
        PROMPT_PAIRS,
        None,
        CONFIG,
        EXECUTION,
        tmp_path / "uninterrupted",
        reports.append,
    )

    interrupted_directory = tmp_path / "interrupted"
    for num_steps in (2, CONFIG.num_steps):
        resumed = run_on_policy_training(
            student,
            teacher,
            PROMPT_PAIRS,
            None,
            replace(CONFIG, num_steps=num_steps),
            EXECUTION,
            interrupted_directory,
            lambda _: None,
        )

    assert [report.metrics.step for report in reports] == [0, 1, 2]
    reverse_kls = [report.metrics.reverse_kl for report in reports]
    assert all(reverse_kl is not None and math.isfinite(reverse_kl) for reverse_kl in reverse_kls)
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


def test_reward_without_teacher_makes_rewarded_responses_more_likely(tmp_path: Path) -> None:
    student = tiny_language_model(seed=0)
    prompt = (5, 9)
    num_rewarded_tokens = student.decoder.vocab_size // 2
    config = replace(
        CONFIG,
        num_steps=12,
        prompts_per_step=2,
        rollouts_per_prompt=8,
        distillation_weight=0.0,
        reward_weight=1.0,
    )

    def reward(_: int, response: Sequence[int]) -> float:
        first_token, *_rest = response
        return float(first_token < num_rewarded_tokens)

    reports: list[StepReport] = []
    trained = run_on_policy_training(
        student,
        None,
        [PromptPair(student=prompt, teacher=prompt)] * (config.num_steps * config.prompts_per_step),
        reward,
        config,
        EXECUTION,
        tmp_path,
        reports.append,
    )

    first_report, *_ = reports
    assert first_report.metrics.reverse_kl is None
    assert first_report.metrics.mean_reward is not None
    assert first_report.rewards is not None
    assert len(first_report.rewards) == config.prompts_per_step * config.rollouts_per_prompt
    # Rollouts of one prompt are sampled independently, otherwise a group could not tell good from bad.
    assert len(set(first_report.responses[: config.rollouts_per_prompt])) > 1
    rewarded_probability_before = jnp.sum(first_token_probabilities(student, prompt)[:num_rewarded_tokens])
    rewarded_probability_after = jnp.sum(first_token_probabilities(trained, prompt)[:num_rewarded_tokens])
    assert rewarded_probability_after > rewarded_probability_before + 0.05


def test_supervised_fine_tuning_lowers_cross_entropy_on_the_given_responses(tmp_path: Path) -> None:
    student = tiny_language_model(seed=0)
    config = SupervisedConfig(
        num_steps=6,
        examples_per_step=4,
        adapter_rank=2,
        learning_rate=1e-2,
        gradient_clip_norm=1.0,
        seed=0,
    )
    example = SupervisedExample(prompt=(5, 9), response=(7, 2, 8, 3))
    reports: list[SupervisedStepReport] = []
    run_supervised_fine_tuning(
        student,
        [example] * (config.num_steps * config.examples_per_step),
        config,
        EXECUTION,
        tmp_path,
        reports.append,
    )

    cross_entropies = [report.metrics.cross_entropy for report in reports]
    first_cross_entropy, *_, last_cross_entropy = cross_entropies
    assert all(math.isfinite(cross_entropy) for cross_entropy in cross_entropies)
    assert last_cross_entropy < first_cross_entropy
    assert reports[0].metrics.finished_fraction == 1.0
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == config.num_steps
