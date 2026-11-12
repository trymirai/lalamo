import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import reduce
from pathlib import Path

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
from .loss import AdvantageBaseline, distillation_advantages, policy_gradient_loss
from .rollouts import Rollouts, realign_response_values, sampled_token_log_probs
from .sampling import ON_POLICY_SAMPLING, pack_rollouts, sample_responses
from .training import add_gradients, apply_gradients, init_optimizer_state, policy_gradients

__all__ = [
    "DistillationConfig",
    "ExecutionConfig",
    "PromptPair",
    "StepMetrics",
    "StepReport",
    "run_on_policy_distillation",
]


@dataclass(frozen=True)
class PromptPair:
    """One conversation as rendered by the chat template of the student and by that of the teacher."""

    student: tuple[int, ...]
    teacher: tuple[int, ...]


@dataclass(frozen=True)
class DistillationConfig:
    num_steps: int
    prompts_per_step: int
    max_response_length: int
    adapter_rank: int
    learning_rate: float
    gradient_clip_norm: float
    baseline: AdvantageBaseline
    seed: int


@dataclass(frozen=True)
class ExecutionConfig:
    gradient_micro_batch_size: int
    student_scoring_micro_batch_size: int
    teacher_scoring_micro_batch_size: int
    checkpoint_every_steps: int


@dataclass(frozen=True)
class StepMetrics:
    step: int
    reverse_kl: float
    student_entropy: float
    teacher_cross_entropy: float
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
    log_prob_gap_quantiles: tuple[float, float, float, float, float]
    strong_disagreement_fraction: float
    response_reverse_kl_quantiles: tuple[float, float, float]
    reverse_kl_by_position: tuple[float | None, ...]
    adapter_norm: float


@dataclass(frozen=True)
class StepReport:
    """A finished step: its metrics, the updated student and the rollouts the step was computed from."""

    metrics: StepMetrics
    student: LanguageModel
    prompt_pairs: Sequence[PromptPair]
    responses: tuple[tuple[int, ...], ...]
    log_prob_gaps: tuple[Float[np.ndarray, " _"], ...]


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


def _move(array: Array, sharding_config: ShardingConfig) -> Array:
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


def _distillation_step(
    student: LanguageModel,
    teacher: LanguageModel,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    prompt_pairs: Sequence[PromptPair],
    config: DistillationConfig,
    execution: ExecutionConfig,
    student_prompt_capacity: int,
    teacher_prompt_capacity: int,
    step: int,
) -> tuple[optax.OptState, StepReport]:
    student_prompts = [pair.student for pair in prompt_pairs]
    teacher_prompts = [pair.teacher for pair in prompt_pairs]
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
    teacher_rollouts = pack_rollouts(
        teacher_prompts,
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
    response_mask = student_rollouts.response_mask
    advantages = distillation_advantages(student_log_probs, teacher_log_probs, response_mask, config.baseline)
    num_response_tokens = jnp.sum(response_mask)
    # Execution is asynchronous: without waiting here, the working memory of the scoring passes is still held
    # when the gradients ask for theirs, and the two peaks add up.
    advantages = jax.block_until_ready(advantages)
    gradient_start = time.monotonic()

    keychain = Keychain.init(config.seed, sharding_config=student.sharding_config)
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
                _micro_batches(student_rollouts, execution.gradient_micro_batch_size),
                _micro_batches(advantages, execution.gradient_micro_batch_size),
                strict=True,
            )
        ),
    )
    decoder, optimizer_state = apply_gradients(student.decoder, optimizer_state, gradients, optimizer)
    gradient_norm = float(optax.global_norm(gradients))
    gradient_end = time.monotonic()

    stop_token_ids = frozenset(student.config.generation_config.stop_token_ids)
    num_tokens = int(num_response_tokens)
    all_log_prob_gaps = np.asarray(teacher_log_probs - student_log_probs)
    # The prediction of the first response token sits one position before it, at the last prompt token.
    log_prob_gaps = tuple(
        row_gaps[len(prompt) - 1 : len(prompt) - 1 + len(response)]
        for row_gaps, prompt, response in zip(all_log_prob_gaps, student_prompts, responses, strict=True)
    )
    flat_log_prob_gaps = np.concatenate(log_prob_gaps)
    response_reverse_kls = [-float(gaps.mean()) for gaps in log_prob_gaps]
    metrics = StepMetrics(
        step=step,
        reverse_kl=float(jnp.sum(student_log_probs - teacher_log_probs)) / num_tokens,
        student_entropy=-float(jnp.sum(student_log_probs)) / num_tokens,
        teacher_cross_entropy=-float(jnp.sum(teacher_log_probs)) / num_tokens,
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
        teacher_peak_memory_gigabytes=_peak_memory_gigabytes(teacher.sharding_config),
        log_prob_gap_quantiles=tuple(np.quantile(flat_log_prob_gaps, (0.01, 0.1, 0.5, 0.9, 0.99)).tolist()),
        strong_disagreement_fraction=float(np.mean(flat_log_prob_gaps < -1.0)),
        response_reverse_kl_quantiles=tuple(np.quantile(response_reverse_kls, (0.0, 0.5, 1.0)).tolist()),
        reverse_kl_by_position=_reverse_kl_by_position(log_prob_gaps, config.max_response_length, 4),
        adapter_norm=_adapter_norm(decoder),
    )
    report = StepReport(
        metrics=metrics,
        student=eqx.tree_at(lambda model: model.decoder, student, decoder),
        prompt_pairs=prompt_pairs,
        responses=responses,
        log_prob_gaps=log_prob_gaps,
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


def run_on_policy_distillation(
    student: LanguageModel,
    teacher: LanguageModel,
    prompt_pairs: Sequence[PromptPair],
    config: DistillationConfig,
    execution: ExecutionConfig,
    output_directory: Path | str,
    on_step: Callable[[StepReport], None],
) -> LanguageModel:
    """Trains low-rank adapters on the student so that its own samples become likely under the teacher.

    Every step samples responses from the current student, scores them with both models and takes one optimizer
    step on the sampled-token estimate of the reverse KL. Step metrics are appended to `metrics.jsonl`. Checkpoints
    are kept under `checkpoints`, and a run restarted with the same `output_directory` continues from the latest.
    """
    num_required_prompts = config.num_steps * config.prompts_per_step
    if len(prompt_pairs) < num_required_prompts:
        raise ValueError(f"The run needs {num_required_prompts} prompts, but only {len(prompt_pairs)} were given.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output_directory / "checkpoints"
    metrics_path = output_directory / "metrics.jsonl"
    student_prompt_capacity = max(len(pair.student) for pair in prompt_pairs)
    teacher_prompt_capacity = max(len(pair.teacher) for pair in prompt_pairs)

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
        step_prompt_pairs = prompt_pairs[step * config.prompts_per_step : (step + 1) * config.prompts_per_step]
        optimizer_state, report = _distillation_step(
            student,
            teacher,
            optimizer_state,
            optimizer,
            step_prompt_pairs,
            config,
            execution,
            student_prompt_capacity,
            teacher_prompt_capacity,
            step,
        )
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
