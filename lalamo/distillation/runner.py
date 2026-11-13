import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import reduce
from pathlib import Path
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PyTree

from lalamo.compressed.low_rank import LowRankMatrix
from lalamo.models import LanguageModel
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules import Decoder, DecoderForwardPassConfig
from lalamo.utils.surgery import select_nodes_of_type

from .adapters import inject_low_rank_adapters, partition_adapters
from .loss import AdvantageBaseline, distillation_advantages, policy_gradient_loss, reward_advantages
from .rollouts import Rollouts, realign_response_values, sampled_token_log_probs
from .sampling import ON_POLICY_SAMPLING, pack_rollouts, sample_responses
from .training import add_gradients, apply_gradients, init_optimizer_state, policy_gradients

__all__ = [
    "ExecutionConfig",
    "OnPolicyConfig",
    "PromptPair",
    "RewardFunction",
    "StepMetrics",
    "StepReport",
    "SupervisedConfig",
    "SupervisedExample",
    "SupervisedStepMetrics",
    "SupervisedStepReport",
    "run_on_policy_training",
    "run_supervised_fine_tuning",
]

# Maps the index of a prompt among the prompts of a run, and a response sampled for it, to the reward of the response.
type RewardFunction = Callable[[int, Sequence[int]], float]


@dataclass(frozen=True)
class PromptPair:
    """One conversation as rendered by the chat template of the student and by that of the teacher."""

    student: tuple[int, ...]
    teacher: tuple[int, ...]


@dataclass(frozen=True)
class OnPolicyConfig:
    """A step samples `rollouts_per_prompt` responses for each of `prompts_per_step` prompts.

    The advantage of a sampled token is `distillation_weight` times its distillation advantage plus `reward_weight`
    times the reward of its rollout, standardized among the rollouts of the same prompt.
    """

    num_steps: int
    prompts_per_step: int
    max_response_length: int
    adapter_rank: int
    learning_rate: float
    gradient_clip_norm: float
    baseline: AdvantageBaseline
    seed: int
    rollouts_per_prompt: int = 1
    distillation_weight: float = 1.0
    reward_weight: float = 0.0


@dataclass(frozen=True)
class SupervisedExample:
    prompt: tuple[int, ...]
    response: tuple[int, ...]


@dataclass(frozen=True)
class SupervisedConfig:
    num_steps: int
    examples_per_step: int
    adapter_rank: int
    learning_rate: float
    gradient_clip_norm: float
    seed: int


@dataclass(frozen=True)
class ExecutionConfig:
    gradient_micro_batch_size: int
    student_scoring_micro_batch_size: int
    teacher_scoring_micro_batch_size: int
    checkpoint_every_steps: int


@dataclass(frozen=True)
class StepMetrics:
    """Metrics of an on-policy step. Those that compare with the teacher are `None` in a run without one."""

    step: int
    reverse_kl: float | None
    student_entropy: float
    teacher_cross_entropy: float | None
    policy_gradient_loss: float
    gradient_norm: float
    num_response_tokens: int
    mean_response_length: float
    max_response_length: int
    finished_fraction: float
    sampling_seconds: float
    teacher_scoring_seconds: float
    student_scoring_seconds: float
    gradient_seconds: float
    student_peak_memory_gigabytes: float | None
    teacher_peak_memory_gigabytes: float | None
    # The log-prob gap of a token is the teacher's log-probability of it minus the student's.
    log_prob_gap_quantiles: tuple[float, float, float, float, float] | None
    strong_disagreement_fraction: float | None
    response_reverse_kl_quantiles: tuple[float, float, float] | None
    reverse_kl_by_position: tuple[float | None, ...] | None
    adapter_norm: float
    # A group is informative if its rollouts did not all earn the same reward. `None` in a run without rewards.
    mean_reward: float | None = None
    informative_group_fraction: float | None = None


@dataclass(frozen=True)
class StepReport:
    """A finished step: its metrics, the updated student and the rollouts the step was computed from."""

    metrics: StepMetrics
    student: LanguageModel
    prompt_pairs: Sequence[PromptPair]
    responses: tuple[tuple[int, ...], ...]
    log_prob_gaps: tuple[Float[np.ndarray, " _"], ...] | None
    rewards: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SupervisedStepMetrics:
    step: int
    cross_entropy: float
    gradient_norm: float
    num_response_tokens: int
    mean_response_length: float
    max_response_length: int
    finished_fraction: float
    student_scoring_seconds: float
    gradient_seconds: float
    student_peak_memory_gigabytes: float | None
    adapter_norm: float


@dataclass(frozen=True)
class SupervisedStepReport:
    metrics: SupervisedStepMetrics
    student: LanguageModel


class _Report(Protocol):
    @property
    def metrics(self) -> StepMetrics | SupervisedStepMetrics: ...

    @property
    def student(self) -> LanguageModel: ...


class _Checkpoint(eqx.Module):
    adapters: Decoder
    optimizer_state: optax.OptState
    num_completed_steps: Int[Array, ""]


def _rows[TreeT: PyTree](batch: TreeT, begin: int, end: int) -> TreeT:
    return jax.tree.map(lambda leaf: leaf[begin:end], batch)


def _micro_batches[TreeT: PyTree](batch: TreeT, size: int) -> list[TreeT]:
    first_leaf, *_ = jax.tree.leaves(batch)
    batch_size, *_ = first_leaf.shape
    if batch_size % size != 0:
        raise ValueError(f"A batch of {batch_size} rows cannot be split into micro-batches of {size}.")
    return [_rows(batch, begin, begin + size) for begin in range(0, batch_size, size)]


def _response_log_probs(
    model: LanguageModel,
    rollouts: Rollouts,
    micro_batch_size: int,
) -> Float[Array, "batch predictions"]:
    keychain = Keychain.init(0, sharding_config=model.sharding_config)
    return jnp.concatenate(
        [
            sampled_token_log_probs(
                model.decoder,
                micro_batch,
                DecoderForwardPassConfig.for_inference(),
                keychain=keychain,
            )
            for micro_batch in _micro_batches(rollouts, micro_batch_size)
        ],
    )


def _move(array: Array | np.ndarray, sharding_config: ShardingConfig) -> Array:
    return jax.device_put(np.asarray(array), sharding_config.make_sharding((None,) * array.ndim))


def _peak_memory_gigabytes(sharding_config: ShardingConfig) -> float | None:
    device_stats = [device.memory_stats() for device in sharding_config.mesh.devices.flat]
    peaks = [stats["peak_bytes_in_use"] for stats in device_stats if stats is not None]
    if not peaks:
        return None
    return max(peaks) / 2**30


def _adapter_norm(decoder: Decoder) -> float:
    """Norm of the adapters' up-projections, which start at zero: how far training has moved the student."""
    up_projections = [adapter.up_projection for _, adapter in select_nodes_of_type(LowRankMatrix, decoder)]
    return float(optax.global_norm(up_projections))


def _reverse_kl_by_position(
    log_prob_gaps: Sequence[Float[np.ndarray, " _"]],
    max_response_length: int,
    num_position_ranges: int,
) -> tuple[float | None, ...]:
    """Reverse KL estimate over equal ranges of response positions; `None` for a range no response reached."""
    range_length = math.ceil(max_response_length / num_position_ranges)
    gaps_by_range = [
        np.concatenate([gaps[range_index * range_length : (range_index + 1) * range_length] for gaps in log_prob_gaps])
        for range_index in range(num_position_ranges)
    ]
    return tuple(None if gaps.size == 0 else -float(gaps.mean()) for gaps in gaps_by_range)


def _take_optimizer_step(
    student: LanguageModel,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    rollouts: Rollouts,
    advantages: Float[Array, "batch predictions"],
    execution: ExecutionConfig,
    seed: int,
) -> tuple[Decoder, optax.OptState, float]:
    """One update on the policy-gradient loss of the rollouts; also returns the norm of the gradient."""
    num_response_tokens = jnp.sum(rollouts.response_mask)
    keychain = Keychain.init(seed, sharding_config=student.sharding_config)
    # Reduced lazily, so that only the running sum and one micro-batch's gradients exist at any time.
    gradients = reduce(
        add_gradients,
        (
            policy_gradients(
                student.decoder,
                rollouts_micro_batch,
                advantages_micro_batch,
                num_response_tokens,
                DecoderForwardPassConfig.for_training(),
                keychain=keychain,
            )
            for rollouts_micro_batch, advantages_micro_batch in zip(
                _micro_batches(rollouts, execution.gradient_micro_batch_size),
                _micro_batches(advantages, execution.gradient_micro_batch_size),
                strict=True,
            )
        ),
    )
    decoder, optimizer_state = apply_gradients(student.decoder, optimizer_state, gradients, optimizer)
    return decoder, optimizer_state, float(optax.global_norm(gradients))


def _on_policy_step(
    student: LanguageModel,
    teacher: LanguageModel | None,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    prompt_pairs: Sequence[PromptPair],
    first_prompt_index: int,
    reward: RewardFunction | None,
    config: OnPolicyConfig,
    execution: ExecutionConfig,
    student_prompt_capacity: int,
    teacher_prompt_capacity: int,
    step: int,
) -> tuple[optax.OptState, StepReport]:
    # The rollouts of one prompt are consecutive, which is the grouping that reward advantages expect.
    prompt_offsets = [offset for offset in range(len(prompt_pairs)) for _ in range(config.rollouts_per_prompt)]
    rollout_prompt_pairs = [prompt_pairs[offset] for offset in prompt_offsets]
    student_prompts = [pair.student for pair in rollout_prompt_pairs]
    sampling_start = time.monotonic()
    responses = sample_responses(
        student,
        student_prompts,
        config.max_response_length,
        ON_POLICY_SAMPLING,
        prompt_capacity=student_prompt_capacity,
        keychain=Keychain.init(config.seed + step + 1, sharding_config=student.sharding_config),
    )
    teacher_scoring_start = time.monotonic()

    student_rollouts = pack_rollouts(
        student_prompts,
        responses,
        num_tokens=student_prompt_capacity + config.max_response_length,
        sharding_config=student.sharding_config,
    )
    response_mask = student_rollouts.response_mask
    teacher_log_probs = None
    if teacher is not None:
        teacher_rollouts = pack_rollouts(
            [pair.teacher for pair in rollout_prompt_pairs],
            responses,
            num_tokens=teacher_prompt_capacity + config.max_response_length,
            sharding_config=teacher.sharding_config,
        )
        teacher_log_probs = realign_response_values(
            _move(
                _response_log_probs(teacher, teacher_rollouts, execution.teacher_scoring_micro_batch_size),
                student.sharding_config,
            ),
            _move(teacher_rollouts.prompt_lengths, student.sharding_config),
            student_rollouts,
        )
    student_scoring_start = time.monotonic()
    student_log_probs = _response_log_probs(student, student_rollouts, execution.student_scoring_micro_batch_size)

    rewards = None
    if reward is not None:
        rewards = tuple(
            reward(first_prompt_index + offset, response)
            for offset, response in zip(prompt_offsets, responses, strict=True)
        )
    advantage_terms = []
    if teacher_log_probs is not None and config.distillation_weight != 0:
        advantage_terms.append(
            config.distillation_weight
            * distillation_advantages(student_log_probs, teacher_log_probs, response_mask, config.baseline),
        )
    if rewards is not None and config.reward_weight != 0:
        advantage_terms.append(
            config.reward_weight
            * reward_advantages(
                _move(np.asarray(rewards, dtype=np.float32), student.sharding_config),
                response_mask,
                config.rollouts_per_prompt,
            ),
        )
    advantages = reduce(jnp.add, advantage_terms)
    num_response_tokens = jnp.sum(response_mask)
    # Execution is asynchronous: without waiting here, the working memory of the scoring passes is still held
    # when the gradients ask for theirs, and the two peaks add up.
    advantages = jax.block_until_ready(advantages)
    gradient_start = time.monotonic()
    decoder, optimizer_state, gradient_norm = _take_optimizer_step(
        student,
        optimizer_state,
        optimizer,
        student_rollouts,
        advantages,
        execution,
        config.seed,
    )
    gradient_end = time.monotonic()

    stop_token_ids = frozenset(student.config.generation_config.stop_token_ids)
    num_tokens = int(num_response_tokens)
    log_prob_gaps = None
    if teacher_log_probs is not None:
        all_log_prob_gaps = np.asarray(teacher_log_probs - student_log_probs)
        # The prediction of the first response token sits one position before it, at the last prompt token.
        log_prob_gaps = tuple(
            row_gaps[len(prompt) - 1 : len(prompt) - 1 + len(response)]
            for row_gaps, prompt, response in zip(all_log_prob_gaps, student_prompts, responses, strict=True)
        )
    flat_log_prob_gaps = None if log_prob_gaps is None else np.concatenate(log_prob_gaps)
    response_reverse_kls = None if log_prob_gaps is None else [-float(gaps.mean()) for gaps in log_prob_gaps]
    reward_groups = (
        None
        if rewards is None
        else [
            rewards[begin : begin + config.rollouts_per_prompt]
            for begin in range(0, len(rewards), config.rollouts_per_prompt)
        ]
    )
    metrics = StepMetrics(
        step=step,
        reverse_kl=(
            None if teacher_log_probs is None else float(jnp.sum(student_log_probs - teacher_log_probs)) / num_tokens
        ),
        student_entropy=-float(jnp.sum(student_log_probs)) / num_tokens,
        teacher_cross_entropy=None if teacher_log_probs is None else -float(jnp.sum(teacher_log_probs)) / num_tokens,
        policy_gradient_loss=float(policy_gradient_loss(student_log_probs, advantages, response_mask)),
        gradient_norm=gradient_norm,
        num_response_tokens=num_tokens,
        mean_response_length=num_tokens / len(responses),
        max_response_length=max(len(response) for response in responses),
        finished_fraction=sum(response[-1] in stop_token_ids for response in responses) / len(responses),
        sampling_seconds=teacher_scoring_start - sampling_start,
        teacher_scoring_seconds=student_scoring_start - teacher_scoring_start,
        student_scoring_seconds=gradient_start - student_scoring_start,
        gradient_seconds=gradient_end - gradient_start,
        student_peak_memory_gigabytes=_peak_memory_gigabytes(student.sharding_config),
        teacher_peak_memory_gigabytes=None if teacher is None else _peak_memory_gigabytes(teacher.sharding_config),
        log_prob_gap_quantiles=(
            None
            if flat_log_prob_gaps is None
            else tuple(np.quantile(flat_log_prob_gaps, (0.01, 0.1, 0.5, 0.9, 0.99)).tolist())
        ),
        strong_disagreement_fraction=(
            None if flat_log_prob_gaps is None else float(np.mean(flat_log_prob_gaps < -1.0))
        ),
        response_reverse_kl_quantiles=(
            None
            if response_reverse_kls is None
            else tuple(np.quantile(response_reverse_kls, (0.0, 0.5, 1.0)).tolist())
        ),
        reverse_kl_by_position=(
            None if log_prob_gaps is None else _reverse_kl_by_position(log_prob_gaps, config.max_response_length, 4)
        ),
        adapter_norm=_adapter_norm(decoder),
        mean_reward=None if rewards is None else sum(rewards) / len(rewards),
        informative_group_fraction=(
            None
            if reward_groups is None
            else sum(min(group) != max(group) for group in reward_groups) / len(reward_groups)
        ),
    )
    report = StepReport(
        metrics=metrics,
        student=eqx.tree_at(lambda model: model.decoder, student, decoder),
        prompt_pairs=rollout_prompt_pairs,
        responses=responses,
        log_prob_gaps=log_prob_gaps,
        rewards=rewards,
    )
    return optimizer_state, report


def _supervised_step(
    student: LanguageModel,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    examples: Sequence[SupervisedExample],
    num_tokens: int,
    config: SupervisedConfig,
    execution: ExecutionConfig,
    step: int,
) -> tuple[optax.OptState, SupervisedStepReport]:
    scoring_start = time.monotonic()
    rollouts = pack_rollouts(
        [example.prompt for example in examples],
        [example.response for example in examples],
        num_tokens=num_tokens,
        sharding_config=student.sharding_config,
    )
    student_log_probs = _response_log_probs(student, rollouts, execution.student_scoring_micro_batch_size)
    response_mask = rollouts.response_mask
    num_response_tokens = int(jnp.sum(response_mask))
    cross_entropy = -float(jnp.sum(student_log_probs)) / num_response_tokens
    # With every advantage equal to one, the policy-gradient loss is the cross-entropy on the given responses.
    advantages = jax.block_until_ready(response_mask.astype(jnp.float32))
    gradient_start = time.monotonic()
    decoder, optimizer_state, gradient_norm = _take_optimizer_step(
        student,
        optimizer_state,
        optimizer,
        rollouts,
        advantages,
        execution,
        config.seed,
    )
    gradient_end = time.monotonic()

    stop_token_ids = frozenset(student.config.generation_config.stop_token_ids)
    metrics = SupervisedStepMetrics(
        step=step,
        cross_entropy=cross_entropy,
        gradient_norm=gradient_norm,
        num_response_tokens=num_response_tokens,
        mean_response_length=num_response_tokens / len(examples),
        max_response_length=max(len(example.response) for example in examples),
        finished_fraction=sum(example.response[-1] in stop_token_ids for example in examples) / len(examples),
        student_scoring_seconds=gradient_start - scoring_start,
        gradient_seconds=gradient_end - gradient_start,
        student_peak_memory_gigabytes=_peak_memory_gigabytes(student.sharding_config),
        adapter_norm=_adapter_norm(decoder),
    )
    report = SupervisedStepReport(
        metrics=metrics,
        student=eqx.tree_at(lambda model: model.decoder, student, decoder),
    )
    return optimizer_state, report


def _save_checkpoint(directory: Path, checkpoint: _Checkpoint) -> None:
    """Checkpoints are never overwritten; one written halfway is never mistaken for a finished one."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{int(checkpoint.num_completed_steps):05d}.eqx"
    unfinished_path = path.with_suffix(".unfinished")
    eqx.tree_serialise_leaves(unfinished_path, checkpoint)
    unfinished_path.replace(path)


def _latest_checkpoint_path(directory: Path) -> Path | None:
    paths = sorted(directory.glob("step_*.eqx"))
    if not paths:
        return None
    *_, latest_path = paths
    return latest_path


def _drop_metrics_after(metrics_path: Path, num_completed_steps: int) -> None:
    """Forgets the steps of an interrupted run that came after its last checkpoint; the resumed run redoes them."""
    if not metrics_path.exists():
        return
    kept_lines = [
        line for line in metrics_path.read_text().splitlines() if json.loads(line)["step"] < num_completed_steps
    ]
    metrics_path.write_text("".join(f"{line}\n" for line in kept_lines))


def _load_checkpoint(path: Path, like: _Checkpoint) -> _Checkpoint:
    loaded = eqx.tree_deserialise_leaves(path, like)
    return jax.tree.map(lambda leaf, reference: jax.device_put(leaf, reference.sharding), loaded, like)


def _train_adapters[ReportT: _Report](
    student: LanguageModel,
    config: OnPolicyConfig | SupervisedConfig,
    execution: ExecutionConfig,
    output_directory: Path | str,
    take_step: Callable[
        [LanguageModel, optax.OptState, optax.GradientTransformation, int],
        tuple[optax.OptState, ReportT],
    ],
    on_step: Callable[[ReportT], None],
) -> LanguageModel:
    """Adds low-rank adapters to the student and trains them with one call of `take_step` per step.

    Step metrics are appended to `metrics.jsonl`. Checkpoints are kept under `checkpoints`, and a run restarted
    with the same `output_directory` continues from the latest.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output_directory / "checkpoints"
    metrics_path = output_directory / "metrics.jsonl"

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adamw(config.learning_rate, weight_decay=0.0),
    )
    decoder = inject_low_rank_adapters(student.decoder, config.adapter_rank, key=jax.random.key(config.seed))
    adapters, frozen = partition_adapters(decoder)
    checkpoint = _Checkpoint(
        adapters=adapters,
        optimizer_state=init_optimizer_state(decoder, optimizer),
        num_completed_steps=jnp.zeros((), dtype=jnp.int32),
    )
    latest_checkpoint_path = _latest_checkpoint_path(checkpoint_directory)
    if latest_checkpoint_path is not None:
        checkpoint = _load_checkpoint(latest_checkpoint_path, checkpoint)
    _drop_metrics_after(metrics_path, int(checkpoint.num_completed_steps))
    student = eqx.tree_at(lambda model: model.decoder, student, eqx.combine(checkpoint.adapters, frozen))
    optimizer_state = checkpoint.optimizer_state

    for step in range(int(checkpoint.num_completed_steps), config.num_steps):
        optimizer_state, report = take_step(student, optimizer_state, optimizer, step)
        student = report.student
        with metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps(asdict(report.metrics)) + "\n")

        num_completed_steps = step + 1
        if num_completed_steps % execution.checkpoint_every_steps == 0 or num_completed_steps == config.num_steps:
            adapters, _ = partition_adapters(student.decoder)
            _save_checkpoint(
                checkpoint_directory,
                _Checkpoint(
                    adapters=adapters,
                    optimizer_state=optimizer_state,
                    num_completed_steps=jnp.asarray(num_completed_steps, dtype=jnp.int32),
                ),
            )
        # Reported after the checkpoint, so that a slow or failing callback cannot lose the step.
        on_step(report)
    return student


def run_on_policy_training(
    student: LanguageModel,
    teacher: LanguageModel | None,
    prompt_pairs: Sequence[PromptPair],
    reward: RewardFunction | None,
    config: OnPolicyConfig,
    execution: ExecutionConfig,
    output_directory: Path | str,
    on_step: Callable[[StepReport], None],
) -> LanguageModel:
    """Trains low-rank adapters on the student from responses that the student samples itself.

    Every step samples responses from the current student and takes one optimizer step on a policy-gradient loss.
    With a teacher, sampled tokens are scored by both models and the loss is the sampled-token estimate of the
    reverse KL, so that the student's own samples become likely under the teacher. With a reward, rollouts that
    score above the other rollouts of their prompt are made more likely. Both terms can be combined.
    """
    if config.distillation_weight != 0 and teacher is None:
        raise ValueError("A non-zero distillation weight needs a teacher.")
    if config.reward_weight != 0 and reward is None:
        raise ValueError("A non-zero reward weight needs a reward function.")
    if config.distillation_weight == 0 and config.reward_weight == 0:
        raise ValueError("Either the distillation weight or the reward weight has to be non-zero.")
    num_required_prompts = config.num_steps * config.prompts_per_step
    if len(prompt_pairs) < num_required_prompts:
        raise ValueError(f"The run needs {num_required_prompts} prompts, but only {len(prompt_pairs)} were given.")
    student_prompt_capacity = max(len(pair.student) for pair in prompt_pairs)
    teacher_prompt_capacity = max(len(pair.teacher) for pair in prompt_pairs)

    def take_step(
        student: LanguageModel,
        optimizer_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        step: int,
    ) -> tuple[optax.OptState, StepReport]:
        first_prompt_index = step * config.prompts_per_step
        return _on_policy_step(
            student,
            teacher,
            optimizer_state,
            optimizer,
            prompt_pairs[first_prompt_index : first_prompt_index + config.prompts_per_step],
            first_prompt_index,
            reward,
            config,
            execution,
            student_prompt_capacity,
            teacher_prompt_capacity,
            step,
        )

    return _train_adapters(student, config, execution, output_directory, take_step, on_step)


def run_supervised_fine_tuning(
    student: LanguageModel,
    examples: Sequence[SupervisedExample],
    config: SupervisedConfig,
    execution: ExecutionConfig,
    output_directory: Path | str,
    on_step: Callable[[SupervisedStepReport], None],
) -> LanguageModel:
    """Trains low-rank adapters on the student with the cross-entropy on given responses.

    Every step takes one optimizer step on the next `examples_per_step` examples; no example is used twice.
    """
    num_required_examples = config.num_steps * config.examples_per_step
    if len(examples) < num_required_examples:
        raise ValueError(f"The run needs {num_required_examples} examples, but only {len(examples)} were given.")
    # Every step is padded to the same length, so that all steps share one compiled gradient.
    num_tokens = max(len(example.prompt) for example in examples) + max(len(example.response) for example in examples)

    def take_step(
        student: LanguageModel,
        optimizer_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        step: int,
    ) -> tuple[optax.OptState, SupervisedStepReport]:
        first_example_index = step * config.examples_per_step
        return _supervised_step(
            student,
            optimizer_state,
            optimizer,
            examples[first_example_index : first_example_index + config.examples_per_step],
            num_tokens,
            config,
            execution,
            step,
        )

    return _train_adapters(student, config, execution, output_directory, take_step, on_step)
