import json
import shutil
import time
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
from jaxtyping import DTypeLike

from lalamo.common import flatten_parameters
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.distillation import (
    DistillBatch,
    DistillParameterSummary,
    DistillTrainConfig,
    DistillTrainingState,
    compute_distill_batch_metrics,
    distill_train_step,
    get_muon_weight_dimension_numbers,
    initialize_distill_optimizer_state,
    initialize_distill_training_state,
    materialize_trainable_module,
    summarize_distill_parameters,
)
from lalamo.models import LanguageModel, LanguageModelConfig
from lalamo.modules.common import iter_parameter_leaves
from lalamo.modules.decoder import Decoder
from lalamo.quantization import QuantizationMode
from lalamo.safetensors import safe_write


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
class ExperimentResult:
    teacher_path: str
    student_path: str
    dataset_path: str
    train_examples: int
    eval_examples: int
    max_sequence_length: int
    batch_size: int
    num_steps: int
    learning_rate: float
    seed: int
    optimizer: OptimizerName
    quantization_mode: str
    distill_config: DistillConfigSnapshot
    parameter_summary: DistillParameterSummary
    initial_eval: EvaluationMetrics
    final_eval: EvaluationMetrics
    compilation_seconds: float
    elapsed_seconds: float
    seconds_per_step: float
    output_model_path: str


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


def _evaluate(
    student: Decoder,
    teacher: Decoder,
    batches: list[DistillBatch],
) -> EvaluationMetrics:
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
    train_examples: Annotated[int, typer.Option()] = 256,
    eval_examples: Annotated[int, typer.Option()] = 64,
    max_sequence_length: Annotated[int, typer.Option()] = 256,
    batch_size: Annotated[int, typer.Option()] = 4,
    num_steps: Annotated[int, typer.Option()] = 25,
    learning_rate: Annotated[float, typer.Option()] = 3e-6,
    optimizer_name: Annotated[OptimizerName, typer.Option()] = OptimizerName.ADAMW,
    quantization_mode: Annotated[QuantizationMode, typer.Option()] = QuantizationMode.UINT4,
    compute_dtype_name: Annotated[ComputeDTypeName, typer.Option()] = ComputeDTypeName.AUTO,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    teacher_model = LanguageModelConfig.load_model(teacher_path)
    student_model = LanguageModelConfig.load_model(student_path)

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
            f"Requested {train_examples + eval_examples} usable sequences, got {len(tokenized)} from {dataset_path}",
        )

    train_sequences = tokenized[:train_examples]
    eval_sequences = tokenized[train_examples : train_examples + eval_examples]
    train_batches = _make_batches(train_sequences, batch_size=batch_size)
    eval_batches = _make_batches(eval_sequences, batch_size=batch_size)

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

    initial_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, distill_config)
    initial_eval = _evaluate(initial_student, teacher_model.model, eval_batches)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    step_iterator = cycle(train_batches)
    train_key = jax.random.key(seed)
    compilation_seconds = 0.0
    start_time: float | None = None

    with history_path.open("w") as history_file:
        for step, batch in enumerate(islice(step_iterator, num_steps), start=1):
            train_key, step_key = jax.random.split(train_key)

            if step == 1:
                warmup_start = time.perf_counter()

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

            if step == 1:
                compilation_seconds = time.perf_counter() - warmup_start
                start_time = time.perf_counter()

            train_metrics = compute_distill_batch_metrics(
                materialize_trainable_module(
                    student_model.model,
                    optimizer_state.training_state,
                    distill_config,
                ),
                teacher_model.model,
                batch,
            )
            history_file.write(
                json.dumps(
                    {
                        "step": step,
                        "train_kl_divergence": float(train_metrics.loss),
                        "valid_tokens": int(train_metrics.valid_tokens),
                    },
                )
                + "\n",
            )

    elapsed_seconds = 0.0 if start_time is None else time.perf_counter() - start_time
    final_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, distill_config)
    final_eval = _evaluate(final_student, teacher_model.model, eval_batches)

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
        train_examples=len(train_sequences),
        eval_examples=len(eval_sequences),
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        num_steps=num_steps,
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
        compilation_seconds=compilation_seconds,
        elapsed_seconds=elapsed_seconds,
        seconds_per_step=0.0 if num_steps <= 1 else elapsed_seconds / (num_steps - 1),
        output_model_path=str(model_output_path),
    )

    with (output_dir / "metrics.json").open("w") as metrics_file:
        json.dump(asdict(result), metrics_file, indent=4)

    print(json.dumps(asdict(result), indent=4))


if __name__ == "__main__":
    typer.run(main)
