import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import cycle, islice
from pathlib import Path
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import typer
from cattrs import Converter
from jaxtyping import Array, DTypeLike

from lalamo.common import flatten_parameters
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.distillation import (
    DistillBatch,
    DistillOptimizerState,
    DistillParameterSummary,
    DistillTrainConfig,
    DistillTrainingState,
    TraceDistillBatch,
    compute_distill_batch_metrics,
    compute_trace_distill_batch_metrics,
    distill_train_step,
    get_muon_weight_dimension_numbers,
    initialize_distill_optimizer_state,
    initialize_distill_training_state,
    make_trace_distill_batch,
    materialize_trainable_module,
    summarize_distill_parameters,
    trace_distill_train_step,
)
from lalamo.models import LanguageModel, LanguageModelConfig
from lalamo.modules.common import iter_parameter_leaves
from lalamo.modules.decoder import Decoder
from lalamo.quantization import QuantizationMode
from lalamo.safetensors import safe_write

_CHECKPOINT_CONVERTER = Converter()


class TrainingMode(StrEnum):
    ONLINE_EXACT = "online_exact"
    TRACE_TOPK = "trace_topk"


class OptimizerName(StrEnum):
    ADAMW = "adamw"
    MUON = "muon"
    SGD = "sgd"


class ComputeDTypeName(StrEnum):
    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


@dataclass(frozen=True)
class EvaluationMetrics:
    kl_divergence: float
    top1_agreement: float
    valid_tokens: int


@dataclass(frozen=True)
class DistillConfigSnapshot:
    master_dtype: str
    compute_dtype: str


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int
    best_eval_kl: float | None
    best_step: int | None
    evaluations_without_improvement: int


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    train_kl_divergence: float
    valid_tokens: int
    eval_kl_divergence: float | None = None
    eval_top1_agreement: float | None = None
    eval_valid_tokens: int | None = None


@dataclass(frozen=True)
class ExperimentResult:
    teacher_path: str
    student_path: str
    dataset_path: str
    training_mode: TrainingMode
    train_examples: int
    eval_examples: int
    max_sequence_length: int
    batch_size: int
    num_steps: int
    completed_steps: int
    learning_rate: float
    seed: int
    optimizer: OptimizerName
    quantization_mode: str
    distill_config: DistillConfigSnapshot
    parameter_summary: DistillParameterSummary
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


def _load_conversations(dataset_path: Path | str, *, num_examples: int, seed: int) -> list[list[HFMessage]]:
    dataset_path = Path(dataset_path)
    dataframe = shuffle_dataset(load_hf_parquet(dataset_path), seed=seed)
    if "conversation" in dataframe.columns:
        column_name = "conversation"
    elif "messages" in dataframe.columns:
        column_name = "messages"
    else:
        raise ValueError(f"{dataset_path} must contain a 'conversation' or 'messages' column")

    conversations = dataframe.get_column(column_name).to_list()[:num_examples]
    return [[HFMessage.from_dict(message) for message in conversation] for conversation in conversations]


def _load_traces(dataset_path: Path | str, *, num_examples: int) -> list[LalamoCompletion]:
    dataset_path = Path(dataset_path)
    with dataset_path.open("rb") as trace_fd:
        return list(islice(LalamoCompletion.deserialize_many(trace_fd), num_examples))


def _tokenize_conversations(
    conversations: list[list[HFMessage]],
    *,
    language_model: LanguageModel,
    max_sequence_length: int,
) -> list[jax.Array]:
    tokenized_sequences: list[jax.Array] = []

    for conversation in conversations:
        messages = [message.as_message() for message in conversation]
        token_ids = language_model.message_processor.tokenize_request(messages)
        if len(token_ids) < 2:
            continue
        tokenized_sequences.append(jnp.array(token_ids[:max_sequence_length], dtype=jnp.int32))

    return tokenized_sequences


def _make_batches(sequences: list[jax.Array], *, batch_size: int) -> list[DistillBatch]:
    batches: list[DistillBatch] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        if not batch_sequences:
            continue

        max_length = max(sequence.shape[0] for sequence in batch_sequences)
        padded = jnp.zeros((len(batch_sequences), max_length), dtype=jnp.int32)
        lengths = jnp.array([sequence.shape[0] for sequence in batch_sequences], dtype=jnp.int32)

        for row, sequence in enumerate(batch_sequences):
            padded = padded.at[row, : sequence.shape[0]].set(sequence)

        batches.append(DistillBatch(token_ids=padded, lengths_without_padding=lengths))

    return batches


def _make_trace_batches(
    traces: Sequence[LalamoCompletion],
    *,
    batch_size: int,
    pad_token_id: int,
) -> list[TraceDistillBatch]:
    return [
        make_trace_distill_batch(traces[start : start + batch_size], pad_token_id=pad_token_id)
        for start in range(0, len(traces), batch_size)
    ]


def _evaluate(student: Decoder, teacher: Decoder, batches: list[DistillBatch]) -> EvaluationMetrics:
    total_kl = 0.0
    total_valid_tokens = 0
    total_matches = 0

    for batch in batches:
        batch_metrics = compute_distill_batch_metrics(student, teacher, batch)
        total_kl += float(batch_metrics.loss) * int(batch_metrics.valid_tokens)
        total_valid_tokens += int(batch_metrics.valid_tokens)
        total_matches += int(batch_metrics.top1_matches)

    return EvaluationMetrics(
        kl_divergence=total_kl / total_valid_tokens,
        top1_agreement=total_matches / total_valid_tokens,
        valid_tokens=total_valid_tokens,
    )


def _evaluate_trace(student: Decoder, batches: list[TraceDistillBatch]) -> EvaluationMetrics:
    total_kl = 0.0
    total_valid_tokens = 0
    total_matches = 0

    for batch in batches:
        batch_metrics = compute_trace_distill_batch_metrics(student, batch)
        total_kl += float(batch_metrics.loss) * int(batch_metrics.valid_tokens)
        total_valid_tokens += int(batch_metrics.valid_tokens)
        total_matches += int(batch_metrics.top1_matches)

    return EvaluationMetrics(
        kl_divergence=total_kl / total_valid_tokens,
        top1_agreement=total_matches / total_valid_tokens,
        valid_tokens=total_valid_tokens,
    )


def _save_materialized_student(
    student_path: Path | str,
    output_path: Path | str,
    student_model: LanguageModel,
    materialized_student: Decoder,
) -> None:
    student_path = Path(student_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(student_path / "config.json", output_path / "config.json")
    shutil.copy2(student_path / "tokenizer.json", output_path / "tokenizer.json")

    with Path(output_path / "model.safetensors").open("wb") as fd:
        updated_model = eqx.tree_at(lambda model: model.model, student_model, materialized_student)
        safe_write(fd, flatten_parameters(updated_model.export_weights()))


def _save_checkpoint(
    checkpoint_dir: Path | str,
    checkpoint: ExperimentCheckpoint,
    metadata: CheckpointMetadata,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(checkpoint_dir / "state.eqx", checkpoint)
    with (checkpoint_dir / "metadata.json").open("w") as metadata_file:
        json.dump(_CHECKPOINT_CONVERTER.unstructure(metadata), metadata_file, indent=4)


def _load_checkpoint(
    checkpoint_dir: Path | str,
    like: ExperimentCheckpoint,
) -> tuple[ExperimentCheckpoint, CheckpointMetadata]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint = eqx.tree_deserialise_leaves(checkpoint_dir / "state.eqx", like)
    with (checkpoint_dir / "metadata.json").open() as metadata_file:
        metadata = _CHECKPOINT_CONVERTER.structure(json.load(metadata_file), CheckpointMetadata)
    return checkpoint, metadata


def _build_optimizer(
    name: OptimizerName,
    learning_rate: float,
    training_state: DistillTrainingState,
) -> optax.GradientTransformation:
    match name:
        case OptimizerName.ADAMW:
            return optax.adamw(learning_rate)
        case OptimizerName.MUON:
            return optax.contrib.muon(
                learning_rate=learning_rate,
                muon_weight_dimension_numbers=get_muon_weight_dimension_numbers(training_state),
            )
        case OptimizerName.SGD:
            return optax.sgd(learning_rate)
        case _:
            raise ValueError(f"Unknown optimizer: {name}")


def _resolve_compute_dtype(name: ComputeDTypeName, native_dtype: DTypeLike) -> DTypeLike:
    match name:
        case ComputeDTypeName.AUTO:
            return native_dtype
        case ComputeDTypeName.BFLOAT16:
            return jnp.bfloat16
        case ComputeDTypeName.FLOAT32:
            return jnp.float32
        case _:
            raise ValueError(f"Unknown compute dtype: {name}")


def main(
    teacher_path: Annotated[Path, typer.Option(exists=True, dir_okay=True, file_okay=False)],
    student_path: Annotated[Path, typer.Option(exists=True, dir_okay=True, file_okay=False)],
    dataset_path: Annotated[Path, typer.Option(exists=True, dir_okay=False, file_okay=True)],
    output_dir: Annotated[Path, typer.Option()],
    training_mode: Annotated[TrainingMode, typer.Option()] = TrainingMode.ONLINE_EXACT,
    train_examples: Annotated[int, typer.Option()] = 256,
    eval_examples: Annotated[int, typer.Option()] = 64,
    max_sequence_length: Annotated[int, typer.Option()] = 256,
    batch_size: Annotated[int, typer.Option()] = 4,
    num_steps: Annotated[int, typer.Option()] = 25,
    learning_rate: Annotated[float, typer.Option()] = 3e-6,
    optimizer_name: Annotated[OptimizerName, typer.Option()] = OptimizerName.ADAMW,
    quantization_mode: Annotated[QuantizationMode, typer.Option()] = QuantizationMode.UINT4,
    compute_dtype_name: Annotated[ComputeDTypeName, typer.Option()] = ComputeDTypeName.AUTO,
    eval_every_steps: Annotated[int, typer.Option()] = 0,
    checkpoint_every_steps: Annotated[int, typer.Option()] = 0,
    early_stop_patience: Annotated[int, typer.Option()] = 0,
    resume_from: Annotated[Path | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    teacher_model = LanguageModelConfig.load_model(teacher_path)
    student_model = LanguageModelConfig.load_model(student_path)

    match training_mode:
        case TrainingMode.ONLINE_EXACT:
            conversations = _load_conversations(
                dataset_path,
                num_examples=train_examples + eval_examples,
                seed=seed,
            )
            tokenized = _tokenize_conversations(
                conversations,
                language_model=student_model,
                max_sequence_length=max_sequence_length,
            )
            if len(tokenized) < train_examples + eval_examples:
                raise ValueError(
                    f"Requested {train_examples + eval_examples} usable sequences, "
                    f"got {len(tokenized)} from {dataset_path}",
                )
            train_items = tokenized[:train_examples]
            eval_items = tokenized[train_examples : train_examples + eval_examples]
            train_batches = _make_batches(train_items, batch_size=batch_size)
            eval_batches = _make_batches(eval_items, batch_size=batch_size)
        case TrainingMode.TRACE_TOPK:
            traces = _load_traces(dataset_path, num_examples=train_examples + eval_examples)
            if len(traces) < train_examples + eval_examples:
                raise ValueError(
                    f"Requested {train_examples + eval_examples} traces, got {len(traces)} from {dataset_path}",
                )
            train_items = traces[:train_examples]
            eval_items = traces[train_examples : train_examples + eval_examples]
            train_batches = _make_trace_batches(train_items, batch_size=batch_size, pad_token_id=0)
            eval_batches = _make_trace_batches(eval_items, batch_size=batch_size, pad_token_id=0)

    distill_config = DistillTrainConfig(
        master_dtype=jnp.float32,
        compute_dtype=_resolve_compute_dtype(compute_dtype_name, student_model.model.activation_precision),
    )
    training_state = initialize_distill_training_state(student_model.model, distill_config)
    optimizer = _build_optimizer(optimizer_name, learning_rate, training_state)
    optimizer_state = initialize_distill_optimizer_state(training_state, optimizer)
    parameter_summary = summarize_distill_parameters(
        iter_parameter_leaves(student_model.model),
        distill_config,
    )

    train_key = jax.random.key(seed)
    completed_steps = 0
    best_eval_kl: float | None = None
    best_step: int | None = None
    evaluations_without_improvement = 0
    resume_best_checkpoint_dir: Path | None = None
    if resume_from is not None:
        checkpoint, checkpoint_metadata = _load_checkpoint(
            resume_from,
            ExperimentCheckpoint(
                optimizer_state=optimizer_state,
                train_key_data=jax.random.key_data(train_key),
            ),
        )
        optimizer_state = checkpoint.optimizer_state
        train_key = jax.random.wrap_key_data(checkpoint.train_key_data)
        completed_steps = checkpoint_metadata.step
        best_eval_kl = checkpoint_metadata.best_eval_kl
        best_step = checkpoint_metadata.best_step
        evaluations_without_improvement = checkpoint_metadata.evaluations_without_improvement
        if best_step is not None and best_step != completed_steps:
            resume_best_checkpoint_dir = Path(resume_from).parent / "best-checkpoint"
            if not resume_best_checkpoint_dir.exists():
                raise ValueError(f"Resume checkpoint {resume_from} is missing sibling best-checkpoint directory")

    initial_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, distill_config)
    match training_mode:
        case TrainingMode.ONLINE_EXACT:
            initial_eval = _evaluate(initial_student, teacher_model.model, eval_batches)
        case TrainingMode.TRACE_TOPK:
            initial_eval = _evaluate_trace(initial_student, eval_batches)

    if best_eval_kl is None:
        best_eval_kl = initial_eval.kl_divergence
        best_step = max(0, completed_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_dir = output_dir / "latest-checkpoint"
    best_checkpoint_dir = output_dir / "best-checkpoint"
    if resume_best_checkpoint_dir is not None:
        shutil.copytree(resume_best_checkpoint_dir, best_checkpoint_dir, dirs_exist_ok=True)
    elif best_step == completed_steps:
        _save_checkpoint(
            best_checkpoint_dir,
            ExperimentCheckpoint(
                optimizer_state=optimizer_state,
                train_key_data=jax.random.key_data(train_key),
            ),
            CheckpointMetadata(
                step=completed_steps,
                best_eval_kl=best_eval_kl,
                best_step=best_step,
                evaluations_without_improvement=evaluations_without_improvement,
            ),
        )
    history_path = output_dir / "history.jsonl"
    step_iterator = cycle(train_batches)
    compilation_seconds = 0.0
    start_time: float | None = None
    executed_steps = 0
    stopped_early = False

    with history_path.open("a" if completed_steps > 0 else "w") as history_file:
        for step, batch in enumerate(islice(step_iterator, completed_steps, num_steps), start=completed_steps + 1):
            train_key, step_key = jax.random.split(train_key)

            if executed_steps == 0:
                warmup_start = time.perf_counter()

            match training_mode:
                case TrainingMode.ONLINE_EXACT:
                    optimizer_state, _ = distill_train_step(
                        optimizer_state,
                        optimizer,
                        student_model.model,
                        teacher_model.model,
                        batch,
                        distill_config,
                        quantization_key=step_key,
                        quantization_mode=quantization_mode,
                    )
                    train_metrics = compute_distill_batch_metrics(
                        materialize_trainable_module(
                            student_model.model,
                            optimizer_state.training_state,
                            distill_config,
                        ),
                        teacher_model.model,
                        batch,
                    )
                case TrainingMode.TRACE_TOPK:
                    optimizer_state, _ = trace_distill_train_step(
                        optimizer_state,
                        optimizer,
                        student_model.model,
                        batch,
                        distill_config,
                        quantization_key=step_key,
                        quantization_mode=quantization_mode,
                    )
                    train_metrics = compute_trace_distill_batch_metrics(
                        materialize_trainable_module(
                            student_model.model,
                            optimizer_state.training_state,
                            distill_config,
                        ),
                        batch,
                    )

            if executed_steps == 0:
                compilation_seconds = time.perf_counter() - warmup_start
                start_time = time.perf_counter()

            history_entry = HistoryEntry(
                step=step,
                train_kl_divergence=float(train_metrics.loss),
                valid_tokens=int(train_metrics.valid_tokens),
            )

            if eval_every_steps > 0 and step % eval_every_steps == 0:
                materialized_student = materialize_trainable_module(
                    student_model.model,
                    optimizer_state.training_state,
                    distill_config,
                )
                match training_mode:
                    case TrainingMode.ONLINE_EXACT:
                        eval_metrics = _evaluate(materialized_student, teacher_model.model, eval_batches)
                    case TrainingMode.TRACE_TOPK:
                        eval_metrics = _evaluate_trace(materialized_student, eval_batches)

                history_entry = HistoryEntry(
                    step=step,
                    train_kl_divergence=float(train_metrics.loss),
                    valid_tokens=int(train_metrics.valid_tokens),
                    eval_kl_divergence=eval_metrics.kl_divergence,
                    eval_top1_agreement=eval_metrics.top1_agreement,
                    eval_valid_tokens=eval_metrics.valid_tokens,
                )
                if best_eval_kl is None or eval_metrics.kl_divergence < best_eval_kl:
                    best_eval_kl = eval_metrics.kl_divergence
                    best_step = step
                    evaluations_without_improvement = 0
                    _save_checkpoint(
                        best_checkpoint_dir,
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
                else:
                    evaluations_without_improvement += 1
                    if early_stop_patience > 0 and evaluations_without_improvement >= early_stop_patience:
                        stopped_early = True

            history_file.write(json.dumps(asdict(history_entry)) + "\n")
            executed_steps += 1
            completed_steps = step

            should_save_checkpoint = checkpoint_every_steps > 0 and step % checkpoint_every_steps == 0
            if should_save_checkpoint or stopped_early:
                _save_checkpoint(
                    latest_checkpoint_dir,
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
            if stopped_early:
                break

    _save_checkpoint(
        latest_checkpoint_dir,
        ExperimentCheckpoint(
            optimizer_state=optimizer_state,
            train_key_data=jax.random.key_data(train_key),
        ),
        CheckpointMetadata(
            step=completed_steps,
            best_eval_kl=best_eval_kl,
            best_step=best_step,
            evaluations_without_improvement=evaluations_without_improvement,
        ),
    )

    elapsed_seconds = 0.0 if start_time is None else time.perf_counter() - start_time
    measured_steps = max(executed_steps - 1, 0)

    if best_checkpoint_dir.exists() and best_step is not None and best_step != completed_steps:
        best_checkpoint, _ = _load_checkpoint(
            best_checkpoint_dir,
            ExperimentCheckpoint(
                optimizer_state=optimizer_state,
                train_key_data=jax.random.key_data(train_key),
            ),
        )
        optimizer_state = best_checkpoint.optimizer_state

    final_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, distill_config)
    match training_mode:
        case TrainingMode.ONLINE_EXACT:
            final_eval = _evaluate(final_student, teacher_model.model, eval_batches)
        case TrainingMode.TRACE_TOPK:
            final_eval = _evaluate_trace(final_student, eval_batches)

    model_output_path = output_dir / "student-distilled"
    export_config = DistillTrainConfig(
        master_dtype=distill_config.master_dtype,
        compute_dtype=student_model.model.activation_precision,
    )
    export_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, export_config)
    _save_materialized_student(
        student_path,
        model_output_path,
        student_model,
        export_student,
    )

    result = ExperimentResult(
        teacher_path=str(teacher_path),
        student_path=str(student_path),
        dataset_path=str(dataset_path),
        training_mode=training_mode,
        train_examples=len(train_items),
        eval_examples=len(eval_items),
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        num_steps=num_steps,
        completed_steps=completed_steps,
        learning_rate=learning_rate,
        seed=seed,
        optimizer=optimizer_name,
        quantization_mode=quantization_mode.value,
        distill_config=DistillConfigSnapshot(
            master_dtype=str(jnp.dtype(distill_config.master_dtype)),
            compute_dtype=str(jnp.dtype(distill_config.compute_dtype)),
        ),
        parameter_summary=parameter_summary,
        initial_eval=initial_eval,
        final_eval=final_eval,
        best_step=best_step,
        stopped_early=stopped_early,
        compilation_seconds=compilation_seconds,
        elapsed_seconds=elapsed_seconds,
        seconds_per_step=0.0 if measured_steps == 0 else elapsed_seconds / measured_steps,
        output_model_path=str(model_output_path),
    )

    with (output_dir / "metrics.json").open("w") as metrics_file:
        json.dump(asdict(result), metrics_file, indent=4)

    print(json.dumps(asdict(result), indent=4))


if __name__ == "__main__":
    typer.run(main)
