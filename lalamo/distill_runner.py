import importlib
import json
import time
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Any, Protocol, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from cattrs import Converter
from jaxtyping import Array, Key

from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.data.lalamo_completions import LalamoCompletion, iter_completions
from lalamo.distillation import (
    DistillBatch,
    DistillBatchMetrics,
    DistillOptimizerState,
    TraceDistillBatch,
    apply_donated_distill_gradients,
    compute_distill_batch_metrics,
    compute_distill_step_gradients,
    compute_trace_distill_batch_metrics,
    compute_trace_distill_step_gradients,
    initialize_distill_training_state,
    inject_lora_adapters,
    make_trace_distill_batch,
    materialize_distill_student,
)
from lalamo.distillation import (
    DistillConfig as DistillTrainConfig,
)
from lalamo.model import Model
from lalamo.models import LanguageModel
from lalamo.module import Keychain
from lalamo.modules.decoder import Decoder
from lalamo.weight_matrix import GradientEstimator

__all__ = [
    "ComputeDTypeName",
    "DistillCallbacks",
    "DistillConfig",
    "DistillResult",
    "EvaluationMetrics",
    "OptimizerName",
    "TrainingMode",
    "distill",
]


_CHECKPOINT_CONVERTER = Converter()


class TrainingMode(StrEnum):
    ONLINE_EXACT = "online_exact"
    TRACE_TOPK = "trace_topk"


class OptimizerName(StrEnum):
    ADAMW = "adamw"
    MUON = "muon"
    SGD = "sgd"


class WandbRun(Protocol):
    summary: MutableMapping[str, object]

    def define_metric(self, name: str, *, step_metric: str | None = None) -> object: ...
    def log(self, data: dict[str, object], *, step: int | None = None) -> object: ...
    def finish(self) -> object: ...


class ComputeDTypeName(StrEnum):
    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


@dataclass(frozen=True)
class EvaluationMetrics:
    kl_divergence: float
    top1_agreement: float
    valid_tokens: int


type DistillBatchLike = DistillBatch | TraceDistillBatch


@dataclass(frozen=True)
class LoadedDistillBatches:
    train_batches: Sequence[DistillBatchLike]
    eval_batches: Sequence[DistillBatchLike]


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int
    best_eval_kl: float | None
    best_step: int | None
    evaluations_without_improvement: int


@dataclass(frozen=True)
class DistillConfig:
    teacher_path: Path
    student_path: Path
    dataset_path: Path
    output_dir: Path
    training_mode: TrainingMode = TrainingMode.ONLINE_EXACT
    train_examples: int = 256
    eval_examples: int = 64
    max_sequence_length: int = 256
    batch_size: int = 4
    num_steps: int = 25
    learning_rate: float = 3e-7
    warmup_steps: int = 0
    gradient_clip_norm: float | None = None
    gradient_accumulation_steps: int = 1
    optimizer_name: OptimizerName = OptimizerName.MUON
    lora_rank: int | None = None
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING
    teacher_argmax_ce_weight: float = 0.0
    hidden_state_mse_weight: float = 0.0
    hard_token_weight: float = 0.0
    compute_dtype_name: ComputeDTypeName = ComputeDTypeName.AUTO
    eval_every_steps: int = 0
    checkpoint_every_steps: int = 0
    early_stop_patience: int = 0
    resume_from: Path | None = None
    seed: int = 0
    save_checkpoints: bool = True
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


@dataclass(frozen=True)
class DistillResult:
    config: DistillConfig
    completed_steps: int
    master_dtype: str
    compute_dtype: str
    initial_eval: EvaluationMetrics
    final_eval: EvaluationMetrics
    best_step: int | None
    stopped_early: bool
    compilation_seconds: float
    elapsed_seconds: float
    seconds_per_step: float
    output_model_path: str


class ExperimentCheckpoint(eqx.Module):
    optimizer_state: DistillOptimizerState
    train_key_data: Array


class DistillCallbacks(Protocol):
    def started(self) -> None: ...
    def loading_models(self) -> None: ...
    def finished_loading_models(self) -> None: ...
    def loading_dataset(self) -> None: ...
    def finished_loading_dataset(self) -> None: ...
    def distillation_progress(self, step: int, num_steps: int) -> None: ...
    def saving_model(self) -> None: ...
    def finished_saving_model(self, output_model_path: Path) -> None: ...
    def finished(self, result: DistillResult) -> None: ...


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------


def _load_tokenized_conversations(
    dataset_path: Path,
    *,
    seed: int,
    model: LanguageModel,
    max_sequence_length: int,
    num_examples: int,
) -> list[np.ndarray]:
    dataframe = shuffle_dataset(load_hf_parquet(dataset_path), seed=seed)
    if "conversation" not in dataframe.columns:
        if "messages" in dataframe.columns:
            dataframe = dataframe.rename({"messages": "conversation"})
        else:
            raise ValueError(f"{dataset_path} must contain a 'conversation' or 'messages' column")

    conversations = dataframe.get_column("conversation").to_list()
    all_token_ids = model.token_codec.encode_requests(
        [HFMessage.from_dict(message).as_message() for message in conversation] for conversation in conversations
    )
    return [
        np.array(ids[:max_sequence_length], dtype=np.int32)
        for ids in all_token_ids
        if len(ids[:max_sequence_length]) >= 2
    ][:num_examples]


def _load_traces(dataset_path: Path, *, num_examples: int) -> list[LalamoCompletion]:
    return list(islice(iter_completions(dataset_path), num_examples))


def _make_batches(
    sequences: list[np.ndarray],
    *,
    batch_size: int,
    fixed_sequence_length: int,
) -> list[DistillBatch]:
    batches: list[DistillBatch] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        padded = np.zeros((len(batch_sequences), fixed_sequence_length), dtype=np.int32)
        lengths = np.array([seq.shape[0] for seq in batch_sequences], dtype=np.int32)
        for row, seq in enumerate(batch_sequences):
            padded[row, : seq.shape[0]] = seq
        batches.append(DistillBatch(token_ids=jnp.array(padded), lengths_without_padding=jnp.array(lengths)))
    return batches


def _load_distill_batches(
    config: DistillConfig,
    model: LanguageModel,
    *,
    batch_size: int,
) -> LoadedDistillBatches:
    requested = config.train_examples + config.eval_examples
    match config.training_mode:
        case TrainingMode.ONLINE_EXACT:
            tokenized = _load_tokenized_conversations(
                config.dataset_path,
                seed=config.seed,
                model=model,
                max_sequence_length=config.max_sequence_length,
                num_examples=requested,
            )
            if len(tokenized) < requested:
                raise ValueError(f"Requested {requested} sequences, got {len(tokenized)} from {config.dataset_path}")
            return LoadedDistillBatches(
                train_batches=_make_batches(
                    tokenized[: config.train_examples],
                    batch_size=batch_size,
                    fixed_sequence_length=config.max_sequence_length,
                ),
                eval_batches=_make_batches(
                    tokenized[config.train_examples : requested],
                    batch_size=batch_size,
                    fixed_sequence_length=config.max_sequence_length,
                ),
            )
        case TrainingMode.TRACE_TOPK:
            traces = _load_traces(config.dataset_path, num_examples=requested)
            if len(traces) < requested:
                raise ValueError(f"Requested {requested} traces, got {len(traces)} from {config.dataset_path}")
            train_traces = traces[: config.train_examples]
            eval_traces = traces[config.train_examples : requested]
            return LoadedDistillBatches(
                train_batches=[
                    make_trace_distill_batch(train_traces[s : s + batch_size], pad_token_id=0)
                    for s in range(0, len(train_traces), batch_size)
                ],
                eval_batches=[
                    make_trace_distill_batch(eval_traces[s : s + batch_size], pad_token_id=0)
                    for s in range(0, len(eval_traces), batch_size)
                ],
            )


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def _evaluate_with(
    training_mode: TrainingMode,
    student: Decoder,
    teacher: Decoder,
    batches: Sequence[DistillBatchLike],
    distill_config: DistillTrainConfig,
    *,
    seed: int,
) -> EvaluationMetrics:
    total_kl = 0.0
    total_valid_tokens = 0
    total_matches = 0

    for i, batch in enumerate(batches):
        batch_keychain = Keychain.init(seed + i)
        match training_mode:
            case TrainingMode.ONLINE_EXACT:
                assert isinstance(batch, DistillBatch)
                metrics = compute_distill_batch_metrics(
                    student,
                    teacher,
                    batch,
                    distill_config,
                    keychain=batch_keychain,
                )
            case TrainingMode.TRACE_TOPK:
                assert isinstance(batch, TraceDistillBatch)
                metrics = compute_trace_distill_batch_metrics(student, batch, distill_config, keychain=batch_keychain)
        total_kl += float(metrics.kl_divergence) * int(metrics.valid_tokens)
        total_valid_tokens += int(metrics.valid_tokens)
        total_matches += int(metrics.top1_matches)

    if total_valid_tokens == 0:
        raise ValueError("Evaluation requires at least one valid token")
    return EvaluationMetrics(
        kl_divergence=total_kl / total_valid_tokens,
        top1_agreement=total_matches / total_valid_tokens,
        valid_tokens=total_valid_tokens,
    )


# ---------------------------------------------------------------------------
#  Optimizer
# ---------------------------------------------------------------------------


def _build_optimizer(
    name: OptimizerName,
    learning_rate: float,
    *,
    warmup_steps: int,
    gradient_clip_norm: float | None,
) -> optax.GradientTransformation:
    schedule = learning_rate if warmup_steps <= 0 else optax.linear_schedule(0.0, learning_rate, warmup_steps)
    match name:
        case OptimizerName.ADAMW:
            optimizer = optax.adamw(schedule)
        case OptimizerName.MUON:
            optimizer = optax.contrib.muon(learning_rate=schedule)
        case OptimizerName.SGD:
            optimizer = optax.sgd(schedule)
        case _:
            raise ValueError(f"Unknown optimizer: {name}")

    if gradient_clip_norm is None:
        return optimizer
    return optax.chain(optax.clip_by_global_norm(gradient_clip_norm), optimizer)


# ---------------------------------------------------------------------------
#  Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(
    checkpoint_dir: Path,
    checkpoint: ExperimentCheckpoint,
    metadata: CheckpointMetadata,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(checkpoint_dir / "state.eqx", checkpoint)
    with (checkpoint_dir / "metadata.json").open("w") as f:
        json.dump(_CHECKPOINT_CONVERTER.unstructure(metadata), f, indent=4)


def _load_checkpoint(
    checkpoint_dir: Path,
    like: ExperimentCheckpoint,
) -> tuple[ExperimentCheckpoint, CheckpointMetadata]:
    checkpoint = eqx.tree_deserialise_leaves(checkpoint_dir / "state.eqx", like)
    with (checkpoint_dir / "metadata.json").open() as f:
        metadata = _CHECKPOINT_CONVERTER.structure(json.load(f), CheckpointMetadata)
    return checkpoint, metadata


def _copy_master_weights(master_weights: Any) -> Any:  # noqa: ANN401
    return jax.tree.map(
        lambda leaf: leaf + jnp.zeros((), dtype=leaf.dtype) if eqx.is_array(leaf) else leaf,
        master_weights,
        is_leaf=lambda leaf: leaf is None,
    )


def _start_wandb_run(config: DistillConfig) -> WandbRun | None:
    if not config.wandb_project:
        return None
    wandb = cast("Any", importlib.import_module("wandb"))
    run = wandb.init(
        entity=config.wandb_entity, project=config.wandb_project,
        name=config.wandb_run_name or config.output_dir.name,
        dir=str(config.output_dir),
        config=json.loads(json.dumps(asdict(config), default=str)),
        settings=wandb.Settings(init_timeout=180),
    )
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    return run


# ---------------------------------------------------------------------------
#  Training step
# ---------------------------------------------------------------------------


def _accumulate_train_step(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    microbatches: Sequence[DistillBatchLike],
    train_key: Key[Array, ""],
    *,
    training_mode: TrainingMode,
    teacher: Decoder,
    distill_config: DistillTrainConfig,
) -> tuple[DistillOptimizerState, DistillBatchMetrics, Key[Array, ""]]:
    accumulated_grads: object | None = None
    total_loss = jnp.zeros((), dtype=jnp.float32)
    total_kl = jnp.zeros((), dtype=jnp.float32)
    total_valid = jnp.zeros((), dtype=jnp.int32)
    total_matches = jnp.zeros((), dtype=jnp.int32)

    for microbatch in microbatches:
        train_key, step_key = jax.random.split(train_key)
        keychain = Keychain.init(int(step_key[0]))
        match training_mode:
            case TrainingMode.ONLINE_EXACT:
                assert isinstance(microbatch, DistillBatch)
                grads, metrics = compute_distill_step_gradients(
                    optimizer_state.training_state,
                    teacher,
                    microbatch,
                    distill_config,
                    keychain=keychain,
                )
            case TrainingMode.TRACE_TOPK:
                assert isinstance(microbatch, TraceDistillBatch)
                grads, metrics = compute_trace_distill_step_gradients(
                    optimizer_state.training_state,
                    microbatch,
                    distill_config,
                    keychain=keychain,
                )
        valid = metrics.valid_tokens
        weighted = jax.tree.map(lambda g, s=valid: g * s, grads)
        accumulated_grads = (
            weighted
            if accumulated_grads is None
            else jax.tree.map(lambda a, b: a + b, accumulated_grads, weighted)
        )
        total_loss = total_loss + metrics.loss * valid.astype(jnp.float32)
        total_kl = total_kl + metrics.kl_divergence * valid.astype(jnp.float32)
        total_valid = total_valid + valid
        total_matches = total_matches + metrics.top1_matches

    assert accumulated_grads is not None
    averaged = jax.tree.map(lambda g: g / total_valid, accumulated_grads)
    step_metrics = DistillBatchMetrics(
        loss=total_loss / total_valid.astype(jnp.float32),
        kl_divergence=total_kl / total_valid.astype(jnp.float32),
        valid_tokens=total_valid,
        top1_matches=total_matches,
    )
    return apply_donated_distill_gradients(optimizer_state, optimizer, averaged), step_metrics, train_key


# ---------------------------------------------------------------------------
#  Main distillation entry point
# ---------------------------------------------------------------------------


def _load_language_model(path: Path) -> LanguageModel:
    model = Model.load(path)
    assert isinstance(model, LanguageModel)
    return model


def distill(
    config: DistillConfig,
    *,
    callbacks: DistillCallbacks | None = None,
) -> DistillResult:
    if config.train_examples < 1:
        raise ValueError("train_examples must be at least 1")
    if config.eval_examples < 1:
        raise ValueError("eval_examples must be at least 1")
    if config.training_mode == TrainingMode.ONLINE_EXACT and config.max_sequence_length < 2:
        raise ValueError("Online distillation requires max_sequence_length >= 2")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _start_wandb_run(config)
    if callbacks is not None:
        callbacks.started()

    # Models
    if callbacks is not None:
        callbacks.loading_models()
    teacher_model = _load_language_model(config.teacher_path)
    student_model = _load_language_model(config.student_path)
    if config.lora_rank is not None:
        student_model = replace(
            student_model,
            decoder=inject_lora_adapters(
                student_model.decoder,
                config.lora_rank,
                key=jax.random.key(config.seed),
            ),
        )
    if callbacks is not None:
        callbacks.finished_loading_models()

    # Dataset
    if callbacks is not None:
        callbacks.loading_dataset()
    dataset = _load_distill_batches(config, student_model, batch_size=config.batch_size)
    if callbacks is not None:
        callbacks.finished_loading_dataset()

    # Training config
    match config.compute_dtype_name:
        case ComputeDTypeName.AUTO:
            compute_dtype = jnp.bfloat16
        case ComputeDTypeName.BFLOAT16:
            compute_dtype = jnp.bfloat16
        case ComputeDTypeName.FLOAT32:
            compute_dtype = jnp.float32
        case _:
            raise ValueError(f"Unknown compute dtype: {config.compute_dtype_name}")

    distill_config = DistillTrainConfig(
        master_dtype=jnp.float32,
        compute_dtype=compute_dtype,
        gradient_estimator=config.gradient_estimator,
        teacher_argmax_ce_weight=config.teacher_argmax_ce_weight,
        hidden_state_mse_weight=config.hidden_state_mse_weight,
        hard_token_weight=config.hard_token_weight,
    )
    training_state = initialize_distill_training_state(
        student_model.decoder,
        distill_config,
        lora=config.lora_rank is not None,
    )
    optimizer = _build_optimizer(
        config.optimizer_name, config.learning_rate,
        warmup_steps=config.warmup_steps, gradient_clip_norm=config.gradient_clip_norm,
    )
    optimizer_state = DistillOptimizerState(
        training_state=training_state,
        optimizer_state=optimizer.init(training_state.master_weights),
    )

    # Checkpoint state
    train_key = jax.random.key(config.seed)
    completed_steps = 0
    best_eval_kl: float | None = None
    best_step: int | None = None
    best_master_weights: Any | None = None
    evaluations_without_improvement = 0

    if config.resume_from is not None:
        checkpoint, meta = _load_checkpoint(
            config.resume_from,
            ExperimentCheckpoint(optimizer_state=optimizer_state, train_key_data=jax.random.key_data(train_key)),
        )
        optimizer_state = checkpoint.optimizer_state
        train_key = jax.random.wrap_key_data(checkpoint.train_key_data)
        completed_steps = meta.step
        best_eval_kl = meta.best_eval_kl
        best_step = meta.best_step
        evaluations_without_improvement = meta.evaluations_without_improvement

    # Initial evaluation
    run_start = time.perf_counter()
    initial_student = materialize_distill_student(optimizer_state.training_state, jnp.dtype(compute_dtype))
    initial_eval = _evaluate_with(
        config.training_mode,
        initial_student,
        teacher_model.decoder,
        dataset.eval_batches,
        distill_config,
        seed=config.seed + 1000,
    )
    if best_eval_kl is None:
        best_eval_kl = initial_eval.kl_divergence
        best_step = completed_steps

    if wandb_run is not None:
        wandb_run.log(
            {
                "step": completed_steps,
                "eval_kl_divergence": initial_eval.kl_divergence,
                "eval_top1_agreement": initial_eval.top1_agreement,
            },
            step=completed_steps,
        )

    latest_checkpoint_dir = config.output_dir / "latest-checkpoint"
    best_checkpoint_dir = config.output_dir / "best-checkpoint"

    def save_checkpoint(checkpoint_dir: Path, step: int) -> None:
        _save_checkpoint(
            checkpoint_dir,
            ExperimentCheckpoint(
                optimizer_state=optimizer_state,
                train_key_data=jax.random.key_data(train_key),
            ),
            CheckpointMetadata(
                step=step,
                best_eval_kl=best_eval_kl,
                best_step=best_step,
                evaluations_without_improvement=evaluations_without_improvement,
            ),
        )

    history_path = config.output_dir / "history.jsonl"
    next_micro = completed_steps * config.gradient_accumulation_steps
    compilation_seconds = 0.0
    executed_steps = 0
    stopped_early = False

    # Training loop
    with history_path.open("a" if completed_steps > 0 else "w") as history:
        for step in range(completed_steps + 1, config.num_steps + 1):
            microbatches = tuple(
                dataset.train_batches[(next_micro + i) % len(dataset.train_batches)]
                for i in range(config.gradient_accumulation_steps)
            )
            next_micro += config.gradient_accumulation_steps

            if executed_steps == 0:
                warmup_start = time.perf_counter()

            optimizer_state, train_metrics, train_key = _accumulate_train_step(
                optimizer_state, optimizer, microbatches, train_key,
                training_mode=config.training_mode,
                teacher=teacher_model.decoder,
                distill_config=distill_config,
            )

            if executed_steps == 0:
                compilation_seconds = time.perf_counter() - warmup_start

            eval_metrics: EvaluationMetrics | None = None
            if config.eval_every_steps > 0 and step % config.eval_every_steps == 0:
                mat_student = materialize_distill_student(
                    optimizer_state.training_state,
                    jnp.dtype(compute_dtype),
                )
                eval_metrics = _evaluate_with(
                    config.training_mode,
                    mat_student,
                    teacher_model.decoder,
                    dataset.eval_batches,
                    distill_config,
                    seed=config.seed + step,
                )
                assert best_eval_kl is not None
                if eval_metrics.kl_divergence < best_eval_kl:
                    best_eval_kl = eval_metrics.kl_divergence
                    best_step = step
                    if not config.save_checkpoints:
                        best_master_weights = _copy_master_weights(optimizer_state.training_state.master_weights)
                    else:
                        save_checkpoint(best_checkpoint_dir, step)
                    evaluations_without_improvement = 0
                else:
                    evaluations_without_improvement += 1
                    if (
                        config.early_stop_patience > 0
                        and evaluations_without_improvement >= config.early_stop_patience
                    ):
                        stopped_early = True

            entry: dict[str, object] = {
                "step": step,
                "train_loss": float(train_metrics.loss),
                "train_kl_divergence": float(train_metrics.kl_divergence),
                "valid_tokens": int(train_metrics.valid_tokens),
                "eval_kl_divergence": None if eval_metrics is None else eval_metrics.kl_divergence,
                "eval_top1_agreement": None if eval_metrics is None else eval_metrics.top1_agreement,
            }
            history.write(json.dumps(entry) + "\n")
            history.flush()
            if wandb_run is not None:
                wandb_run.log(entry, step=step)
            executed_steps += 1
            completed_steps = step
            if callbacks is not None:
                callbacks.distillation_progress(step, config.num_steps)

            should_save_checkpoint = (
                config.checkpoint_every_steps > 0 and step % config.checkpoint_every_steps == 0
            ) or stopped_early
            if config.save_checkpoints and should_save_checkpoint:
                save_checkpoint(latest_checkpoint_dir, step)
            if stopped_early:
                break

    if config.save_checkpoints:
        save_checkpoint(latest_checkpoint_dir, completed_steps)

    # Restore best if needed
    if best_step != completed_steps and config.eval_every_steps > 0:
        if config.save_checkpoints and best_checkpoint_dir.exists():
            best_ckpt, _ = _load_checkpoint(
                best_checkpoint_dir,
                ExperimentCheckpoint(
                    optimizer_state=optimizer_state,
                    train_key_data=jax.random.key_data(train_key),
                ),
            )
            optimizer_state = best_ckpt.optimizer_state
        elif best_master_weights is not None:
            optimizer_state = replace(
                optimizer_state,
                training_state=replace(optimizer_state.training_state, master_weights=best_master_weights),
            )

    # Final evaluation and export
    final_student = materialize_distill_student(optimizer_state.training_state, jnp.dtype(compute_dtype))
    final_eval = _evaluate_with(
        config.training_mode,
        final_student,
        teacher_model.decoder,
        dataset.eval_batches,
        distill_config,
        seed=config.seed + 9999,
    )

    model_output_path = config.output_dir / "student-distilled"
    if callbacks is not None:
        callbacks.saving_model()
    export_student = materialize_distill_student(
        optimizer_state.training_state,
        jnp.bfloat16,
    )
    updated_model = replace(student_model, decoder=export_student)
    updated_model.save(model_output_path)
    if callbacks is not None:
        callbacks.finished_saving_model(model_output_path)

    elapsed = time.perf_counter() - run_start
    measured = max(executed_steps - 1, 0)
    result = DistillResult(
        config=config,
        completed_steps=completed_steps,
        master_dtype=str(jnp.dtype(jnp.float32)),
        compute_dtype=str(jnp.dtype(compute_dtype)),
        initial_eval=initial_eval,
        final_eval=final_eval,
        best_step=best_step,
        stopped_early=stopped_early,
        compilation_seconds=compilation_seconds,
        elapsed_seconds=elapsed,
        seconds_per_step=0.0 if measured == 0 else elapsed / measured,
        output_model_path=str(model_output_path),
    )
    with (config.output_dir / "metrics.json").open("w") as f:
        json.dump(asdict(result), f, indent=4, default=str)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best_step": best_step,
                "final_eval_kl": final_eval.kl_divergence,
                "elapsed_seconds": elapsed,
            }
        )
        wandb_run.finish()
    if callbacks is not None:
        callbacks.finished(result)
    return result
